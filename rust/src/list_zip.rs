use std::collections::HashSet;

use arrow::array::builder::{make_builder, ShareStrategy};
use arrow::array::{Array, StructArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use arrow::legacy::prelude::LargeListArray;
use arrow::offset::Offsets;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
struct ZipKwargs {
    pad: bool,
    #[serde(default)]
    broadcast: bool,
    fields: Option<Vec<String>>,
}

fn list_zip_output(input_fields: &[Field], kwargs: ZipKwargs) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() >= 2,
        ComputeError: "zip_list expects at least 2 columns, got {}",
        input_fields.len()
    );

    polars_ensure!(
        !kwargs.broadcast || input_fields.iter().any(|field| matches!(field.dtype(), DataType::List(_))),
        SchemaMismatch: "broadcast requires at least one List input"
    );
    let field_names = normalized_field_names(input_fields, kwargs.fields.as_deref())?;
    let mut struct_fields = Vec::with_capacity(input_fields.len());

    for (input_field, field_name) in input_fields.iter().zip(field_names) {
        let inner_dtype = match input_field.dtype() {
            DataType::List(inner) => inner.as_ref().clone(),
            dtype if kwargs.broadcast => dtype.clone(),
            dtype => {
                polars_bail!(
                    SchemaMismatch:
                    "invalid series dtype: expected `List`, got `{}`",
                    dtype
                )
            }
        };
        struct_fields.push(Field::new(field_name.into(), inner_dtype));
    }

    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Struct(struct_fields))),
    ))
}

#[polars_expr(output_type_func_with_kwargs=list_zip_output)]
fn list_zip(inputs: &[Series], kwargs: ZipKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() >= 2,
        ComputeError: "zip_list expects at least 2 columns, got {}",
        inputs.len()
    );

    let input_fields: Vec<Field> = inputs
        .iter()
        .map(|series| Field::new(series.name().clone(), series.dtype().clone()))
        .collect();
    let field_names = normalized_field_names(&input_fields, kwargs.fields.as_deref())?;
    let output_dtype = list_zip_output(&input_fields, kwargs.clone())?
        .dtype()
        .clone();
    let prepared = prepare_inputs(inputs)?;
    let arrays: Vec<&dyn Array> = prepared.iter().map(|s| s.chunks()[0].as_ref()).collect();
    let lists: Vec<Option<&LargeListArray>> = arrays
        .iter()
        .zip(prepared.iter())
        .map(|(array, series)| {
            if matches!(series.dtype(), DataType::List(_)) {
                Some(array.as_any().downcast_ref::<LargeListArray>().unwrap())
            } else {
                None
            }
        })
        .collect();

    let rows = prepared[0].len();
    let total_capacity = total_struct_values(&lists, rows, kwargs.pad);
    let mut child_builders = arrays
        .iter()
        .zip(lists.iter())
        .map(|(arr, list)| {
            let mut builder = make_builder(list.map_or(arr.dtype(), |list| list.values().dtype()));
            builder.reserve(total_capacity);
            builder
        })
        .collect::<Vec<_>>();
    let mut validity = MutableBitmap::with_capacity(rows);
    let mut offsets = Offsets::<i64>::with_capacity(rows);

    for row in 0..rows {
        if lists.iter().flatten().any(|arr| !arr.is_valid(row)) {
            validity.push(false);
            offsets.try_push(0)?;
            continue;
        }

        validity.push(true);
        let lengths = lists
            .iter()
            .flatten()
            .map(|arr| arr.offsets().length_at(row));
        let target_len = if kwargs.pad {
            lengths.max()
        } else {
            lengths.min()
        }
        .unwrap_or(0);
        for ((arr, list), builder) in arrays
            .iter()
            .zip(lists.iter())
            .zip(child_builders.iter_mut())
        {
            if let Some(list) = list {
                let source_len = list.offsets().length_at(row);
                let source_start = list.offsets()[row] as usize;
                let copy_len = source_len.min(target_len);
                if copy_len > 0 {
                    builder.subslice_extend(
                        list.values().as_ref(),
                        source_start,
                        copy_len,
                        ShareStrategy::Always,
                    );
                }
                if source_len < target_len {
                    builder.extend_nulls(target_len - source_len);
                }
            } else {
                for _ in 0..target_len {
                    builder.subslice_extend(*arr, row, 1, ShareStrategy::Always);
                }
            }
        }

        offsets.try_push(target_len)?;
    }

    let children = child_builders
        .iter_mut()
        .map(|builder| builder.freeze_reset())
        .collect::<Vec<_>>();
    let arrow_fields = field_names
        .into_iter()
        .zip(children.iter())
        .map(|(name, child)| ArrowField::new(name.into(), child.dtype().clone(), true))
        .collect::<Vec<_>>();
    let struct_dtype = ArrowDataType::Struct(arrow_fields);
    let struct_len = children.first().map_or(0, |child| child.len());
    let struct_arr = StructArray::new(struct_dtype.clone(), struct_len, children, None);
    let list_dtype = LargeListArray::default_datatype(struct_dtype);
    let list_arr = LargeListArray::new(
        list_dtype,
        offsets.into(),
        Box::new(struct_arr),
        validity.into(),
    );
    let output = unsafe {
        ListChunked::from_chunks_and_dtype(
            inputs[0].name().clone(),
            vec![Box::new(list_arr)],
            output_dtype,
        )
    };

    Ok(output.into_series())
}

fn normalized_field_names(
    input_fields: &[Field],
    fields: Option<&[String]>,
) -> PolarsResult<Vec<String>> {
    let names = match fields {
        Some(fields) => {
            polars_ensure!(
                fields.len() == input_fields.len(),
                ComputeError:
                "fields must contain exactly {} names",
                input_fields.len()
            );
            fields.to_vec()
        }
        None => input_fields
            .iter()
            .map(|field| field.name().as_str().to_string())
            .collect(),
    };

    let mut seen = HashSet::with_capacity(names.len());
    for name in &names {
        polars_ensure!(
            seen.insert(name.as_str()),
            Duplicate: "column with name '{}' has more than one occurrence",
            name
        );
    }

    Ok(names)
}

fn prepare_inputs(inputs: &[Series]) -> PolarsResult<Vec<Series>> {
    let target_len = inputs
        .iter()
        .map(|series| series.len())
        .filter(|len| *len != 1)
        .max()
        .unwrap_or(1);
    let mut out = Vec::with_capacity(inputs.len());

    for input in inputs {
        let ca = input;
        polars_ensure!(
            ca.len() == target_len || ca.len() == 1,
            ShapeMismatch:
            "series length {} does not match expected length of {}",
            ca.len(),
            target_len
        );

        let prepared = if ca.len() == target_len {
            ca.clone()
        } else {
            ca.new_from_index(0, target_len)
        };
        out.push(prepared.rechunk());
    }

    Ok(out)
}

fn total_struct_values(arrays: &[Option<&LargeListArray>], rows: usize, pad: bool) -> usize {
    if arrays.is_empty() {
        return 0;
    }

    (0..rows)
        .map(|row| {
            if arrays.iter().flatten().any(|arr| !arr.is_valid(row)) {
                0
            } else if pad {
                arrays
                    .iter()
                    .flatten()
                    .map(|arr| arr.offsets().length_at(row))
                    .max()
                    .unwrap_or(0)
            } else {
                arrays
                    .iter()
                    .flatten()
                    .map(|arr| arr.offsets().length_at(row))
                    .min()
                    .unwrap_or(0)
            }
        })
        .sum()
}
