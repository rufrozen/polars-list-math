use std::borrow::Cow;
use std::collections::HashSet;

use arrow::array::builder::{make_builder, ShareStrategy};
use arrow::array::{Array, StructArray, UInt32Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use arrow::legacy::prelude::LargeListArray;
use arrow::offset::Offsets;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
struct CombinationsKwargs {
    left_index: Option<String>,
    right_index: Option<String>,
    left_value: String,
    right_value: String,
    with_index: bool,
    skip_null: bool,
}

struct CombinationFields {
    include_index: bool,
    left_index: String,
    right_index: String,
    left_value: String,
    right_value: String,
}

const DEFAULT_LEFT_INDEX: &str = "left_index";
const DEFAULT_RIGHT_INDEX: &str = "right_index";

fn list_combinations_output(
    input_fields: &[Field],
    kwargs: CombinationsKwargs,
) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 1,
        ComputeError: "list.combinations expects 1 list column, got {}",
        input_fields.len()
    );

    let value_dtype = list_inner_dtype(input_fields[0].dtype())?;
    combinations_output_field(
        input_fields[0].name().clone(),
        value_dtype,
        value_dtype,
        &kwargs,
    )
}

fn list_combinations_to_output(
    input_fields: &[Field],
    kwargs: CombinationsKwargs,
) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 2,
        ComputeError: "list.combinations_to expects 2 list columns, got {}",
        input_fields.len()
    );

    let left_dtype = list_inner_dtype(input_fields[0].dtype())?;
    let right_dtype = list_inner_dtype(input_fields[1].dtype())?;
    combinations_output_field(
        input_fields[0].name().clone(),
        left_dtype,
        right_dtype,
        &kwargs,
    )
}

#[polars_expr(output_type_func_with_kwargs=list_combinations_output)]
fn list_combinations(inputs: &[Series], kwargs: CombinationsKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "list.combinations expects 1 list column, got {}",
        inputs.len()
    );

    let input_fields = fields_from_series(inputs);
    let output_dtype = list_combinations_output(&input_fields, kwargs.clone())?
        .dtype()
        .clone();
    let fields = combination_fields(&kwargs)?;
    let prepared = prepare_inputs(inputs)?;
    let list = prepared[0].downcast_as_array();
    let total_capacity = total_self_pairs(list);

    let mut left_indices = Vec::<u32>::with_capacity(total_capacity);
    let mut right_indices = Vec::<u32>::with_capacity(total_capacity);
    let mut left_values = make_builder(list.values().dtype());
    let mut right_values = make_builder(list.values().dtype());
    left_values.reserve(total_capacity);
    right_values.reserve(total_capacity);

    let mut validity = MutableBitmap::with_capacity(list.len());
    let mut offsets = Offsets::<i64>::with_capacity(list.len());

    for row in 0..list.len() {
        if !list.is_valid(row) {
            validity.push(false);
            offsets.try_push(0)?;
            continue;
        }

        validity.push(true);
        let start = list.offsets()[row] as usize;
        let length = list.offsets().length_at(row);
        let mut row_count = 0usize;

        for left_index in 0..length {
            for right_index in left_index..length {
                row_count += usize::from(push_pair(
                    &mut left_indices,
                    &mut right_indices,
                    left_values.as_mut(),
                    right_values.as_mut(),
                    list.values().as_ref(),
                    list.values().as_ref(),
                    start + left_index,
                    start + right_index,
                    left_index,
                    right_index,
                    kwargs.skip_null,
                )?);
            }
        }

        offsets.try_push(row_count)?;
    }

    build_output_series(
        inputs[0].name().clone(),
        output_dtype,
        fields,
        left_indices,
        right_indices,
        left_values.freeze_reset(),
        right_values.freeze_reset(),
        offsets,
        validity,
    )
}

#[polars_expr(output_type_func_with_kwargs=list_combinations_to_output)]
fn list_combinations_to(inputs: &[Series], kwargs: CombinationsKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "list.combinations_to expects 2 list columns, got {}",
        inputs.len()
    );

    let input_fields = fields_from_series(inputs);
    let output_dtype = list_combinations_to_output(&input_fields, kwargs.clone())?
        .dtype()
        .clone();
    let fields = combination_fields(&kwargs)?;
    let prepared = prepare_inputs(inputs)?;
    let left = prepared[0].downcast_as_array();
    let right = prepared[1].downcast_as_array();
    let total_capacity = total_cross_pairs(left, right);

    let mut left_indices = Vec::<u32>::with_capacity(total_capacity);
    let mut right_indices = Vec::<u32>::with_capacity(total_capacity);
    let mut left_values = make_builder(left.values().dtype());
    let mut right_values = make_builder(right.values().dtype());
    left_values.reserve(total_capacity);
    right_values.reserve(total_capacity);

    let mut validity = MutableBitmap::with_capacity(left.len());
    let mut offsets = Offsets::<i64>::with_capacity(left.len());

    for row in 0..left.len() {
        if !left.is_valid(row) || !right.is_valid(row) {
            validity.push(false);
            offsets.try_push(0)?;
            continue;
        }

        validity.push(true);
        let left_start = left.offsets()[row] as usize;
        let right_start = right.offsets()[row] as usize;
        let left_length = left.offsets().length_at(row);
        let right_length = right.offsets().length_at(row);
        let mut row_count = 0usize;

        for left_index in 0..left_length {
            for right_index in 0..right_length {
                row_count += usize::from(push_pair(
                    &mut left_indices,
                    &mut right_indices,
                    left_values.as_mut(),
                    right_values.as_mut(),
                    left.values().as_ref(),
                    right.values().as_ref(),
                    left_start + left_index,
                    right_start + right_index,
                    left_index,
                    right_index,
                    kwargs.skip_null,
                )?);
            }
        }

        offsets.try_push(row_count)?;
    }

    build_output_series(
        inputs[0].name().clone(),
        output_dtype,
        fields,
        left_indices,
        right_indices,
        left_values.freeze_reset(),
        right_values.freeze_reset(),
        offsets,
        validity,
    )
}

fn push_pair(
    left_indices: &mut Vec<u32>,
    right_indices: &mut Vec<u32>,
    left_value_builder: &mut dyn arrow::array::builder::ArrayBuilder,
    right_value_builder: &mut dyn arrow::array::builder::ArrayBuilder,
    left_values: &dyn Array,
    right_values: &dyn Array,
    left_absolute_index: usize,
    right_absolute_index: usize,
    left_row_index: usize,
    right_row_index: usize,
    skip_null: bool,
) -> PolarsResult<bool> {
    if skip_null
        && (!left_values.is_valid(left_absolute_index)
            || !right_values.is_valid(right_absolute_index))
    {
        return Ok(false);
    }

    left_indices.push(index_to_u32(left_row_index)?);
    right_indices.push(index_to_u32(right_row_index)?);
    left_value_builder.subslice_extend(left_values, left_absolute_index, 1, ShareStrategy::Always);
    right_value_builder.subslice_extend(
        right_values,
        right_absolute_index,
        1,
        ShareStrategy::Always,
    );
    Ok(true)
}

fn build_output_series(
    name: PlSmallStr,
    output_dtype: DataType,
    fields: CombinationFields,
    left_indices: Vec<u32>,
    right_indices: Vec<u32>,
    left_values: Box<dyn Array>,
    right_values: Box<dyn Array>,
    offsets: Offsets<i64>,
    validity: MutableBitmap,
) -> PolarsResult<Series> {
    let mut named_children =
        Vec::<(String, Box<dyn Array>)>::with_capacity(if fields.include_index { 4 } else { 2 });
    if fields.include_index {
        named_children.push((
            fields.left_index,
            Box::new(UInt32Array::from_vec(left_indices)) as Box<dyn Array>,
        ));
        named_children.push((
            fields.right_index,
            Box::new(UInt32Array::from_vec(right_indices)) as Box<dyn Array>,
        ));
    }
    named_children.push((fields.left_value, left_values));
    named_children.push((fields.right_value, right_values));

    let struct_len = named_children.first().map_or(0, |(_, child)| child.len());
    let mut arrow_fields = Vec::with_capacity(named_children.len());
    let mut children = Vec::with_capacity(named_children.len());
    for (field_name, child) in named_children {
        arrow_fields.push(ArrowField::new(
            field_name.into(),
            child.dtype().clone(),
            true,
        ));
        children.push(child);
    }

    let struct_dtype = ArrowDataType::Struct(arrow_fields);
    let struct_arr = StructArray::new(struct_dtype.clone(), struct_len, children, None);
    let list_dtype = LargeListArray::default_datatype(struct_dtype);
    let list_arr = LargeListArray::new(
        list_dtype,
        offsets.into(),
        Box::new(struct_arr),
        validity.into(),
    );
    let output =
        unsafe { ListChunked::from_chunks_and_dtype(name, vec![Box::new(list_arr)], output_dtype) };

    Ok(output.into_series())
}

fn combinations_output_field(
    name: PlSmallStr,
    left_dtype: &DataType,
    right_dtype: &DataType,
    kwargs: &CombinationsKwargs,
) -> PolarsResult<Field> {
    let fields = combination_fields(kwargs)?;
    let mut struct_fields = Vec::with_capacity(if fields.include_index { 4 } else { 2 });
    if fields.include_index {
        struct_fields.push(Field::new(fields.left_index.into(), DataType::UInt32));
        struct_fields.push(Field::new(fields.right_index.into(), DataType::UInt32));
    }
    struct_fields.push(Field::new(fields.left_value.into(), left_dtype.clone()));
    struct_fields.push(Field::new(fields.right_value.into(), right_dtype.clone()));

    Ok(Field::new(
        name,
        DataType::List(Box::new(DataType::Struct(struct_fields))),
    ))
}

fn combination_fields(kwargs: &CombinationsKwargs) -> PolarsResult<CombinationFields> {
    let include_index =
        kwargs.with_index || kwargs.left_index.is_some() || kwargs.right_index.is_some();
    let left_index = kwargs
        .left_index
        .clone()
        .unwrap_or_else(|| DEFAULT_LEFT_INDEX.to_string());
    let right_index = kwargs
        .right_index
        .clone()
        .unwrap_or_else(|| DEFAULT_RIGHT_INDEX.to_string());

    let mut names = Vec::with_capacity(if include_index { 4 } else { 2 });
    if include_index {
        names.push(left_index.clone());
        names.push(right_index.clone());
    }
    names.push(kwargs.left_value.clone());
    names.push(kwargs.right_value.clone());

    let mut seen = HashSet::with_capacity(names.len());
    for name in &names {
        polars_ensure!(
            seen.insert(name.as_str()),
            Duplicate: "field with name '{}' has more than one occurrence",
            name
        );
    }

    Ok(CombinationFields {
        include_index,
        left_index,
        right_index,
        left_value: kwargs.left_value.clone(),
        right_value: kwargs.right_value.clone(),
    })
}

fn fields_from_series(inputs: &[Series]) -> Vec<Field> {
    inputs
        .iter()
        .map(|series| Field::new(series.name().clone(), series.dtype().clone()))
        .collect()
}

fn list_inner_dtype(dtype: &DataType) -> PolarsResult<&DataType> {
    match dtype {
        DataType::List(inner) => Ok(inner.as_ref()),
        dtype => {
            polars_bail!(
                SchemaMismatch:
                "invalid series dtype: expected `List`, got `{}`",
                dtype
            )
        }
    }
}

fn prepare_inputs(inputs: &[Series]) -> PolarsResult<Vec<ListChunked>> {
    let target_len = inputs.iter().map(|series| series.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(inputs.len());

    for input in inputs {
        let ca = input.list()?;
        polars_ensure!(
            ca.len() == target_len || ca.len() == 1,
            ShapeMismatch:
            "series length {} does not match expected length of {}",
            ca.len(),
            target_len
        );

        let prepared = if ca.len() == target_len {
            Cow::Borrowed(ca)
        } else {
            Cow::Owned(ca.new_from_index(0, target_len))
        };
        out.push(prepared.rechunk().into_owned());
    }

    Ok(out)
}

fn total_self_pairs(list: &LargeListArray) -> usize {
    (0..list.len())
        .map(|row| {
            if list.is_valid(row) {
                let length = list.offsets().length_at(row);
                length.saturating_mul(length.saturating_add(1)) / 2
            } else {
                0
            }
        })
        .sum()
}

fn total_cross_pairs(left: &LargeListArray, right: &LargeListArray) -> usize {
    (0..left.len())
        .map(|row| {
            if left.is_valid(row) && right.is_valid(row) {
                left.offsets()
                    .length_at(row)
                    .saturating_mul(right.offsets().length_at(row))
            } else {
                0
            }
        })
        .sum()
}

fn index_to_u32(index: usize) -> PolarsResult<u32> {
    polars_ensure!(
        index <= u32::MAX as usize,
        ComputeError: "list index {} exceeds UInt32 range",
        index
    );
    Ok(index as u32)
}
