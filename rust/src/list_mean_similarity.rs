use std::ops::Range;

use arrow::array::Array;
use arrow::legacy::prelude::LargeListArray;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::list_similarity_core::{
    document_dtype as document_dtype_from_dtype, list_inner_dtype, prepare_list_inputs,
    validate_document_dtypes, validate_p, weighted_jaccard, DocumentDtype, DocumentValues,
    DocumentWeights,
};

#[derive(Clone, Debug, Deserialize)]
struct ListMeanSimilarityKwargs {
    p: f64,
}

struct ListCollection<'a> {
    outer: &'a LargeListArray,
    inner: &'a LargeListArray,
    values: DocumentValues<'a>,
}

fn list_mean_similarity_output(
    input_fields: &[Field],
    kwargs: ListMeanSimilarityKwargs,
) -> PolarsResult<Field> {
    validate_p(kwargs.p)?;
    validate_self_inputs(input_fields)?;

    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

fn list_mean_similarity_to_output(
    input_fields: &[Field],
    kwargs: ListMeanSimilarityKwargs,
) -> PolarsResult<Field> {
    validate_p(kwargs.p)?;
    validate_cross_inputs(input_fields)?;

    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

#[polars_expr(output_type_func_with_kwargs=list_mean_similarity_output)]
fn list_mean_similarity(
    inputs: &[Series],
    kwargs: ListMeanSimilarityKwargs,
) -> PolarsResult<Series> {
    validate_p(kwargs.p)?;
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "list_mean_similarity expects exactly 1 column, got {}",
        inputs.len()
    );

    let input_fields = fields_from_series(inputs);
    validate_self_inputs(&input_fields)?;

    let prepared = prepare_list_inputs(inputs)?;
    let lists = ListCollection::try_new(&prepared[0], "list_mean_similarity")?;
    let rows = prepared[0].len();
    let mut builder = output_builder(inputs[0].name().clone(), rows, &lists);

    for row in 0..rows {
        match self_row_scores(&lists, row, kwargs.p)? {
            Some(scores) => builder.append_iter(scores.into_iter()),
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type_func_with_kwargs=list_mean_similarity_to_output)]
fn list_mean_similarity_to(
    inputs: &[Series],
    kwargs: ListMeanSimilarityKwargs,
) -> PolarsResult<Series> {
    validate_p(kwargs.p)?;
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "list_mean_similarity_to expects exactly 2 columns, got {}",
        inputs.len()
    );

    let input_fields = fields_from_series(inputs);
    validate_cross_inputs(&input_fields)?;

    let prepared = prepare_list_inputs(inputs)?;
    let target = ListCollection::try_new(&prepared[0], "list_mean_similarity_to")?;
    let reference = ListCollection::try_new(&prepared[1], "list_mean_similarity_to")?;
    let rows = prepared[0].len();
    let mut builder = output_builder(inputs[0].name().clone(), rows, &target);

    for row in 0..rows {
        match cross_row_scores(&target, &reference, row, kwargs.p)? {
            Some(scores) => builder.append_iter(scores.into_iter()),
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

impl<'a> ListCollection<'a> {
    fn try_new(ca: &'a ListChunked, function_name: &str) -> PolarsResult<Self> {
        let outer = ca.downcast_as_array();
        let Some(inner) = outer.values().as_any().downcast_ref::<LargeListArray>() else {
            polars_bail!(
                SchemaMismatch:
                "{} expects dtype List(List(primitive)), got Arrow dtype {:?}",
                function_name,
                outer.values().dtype()
            );
        };

        let values = DocumentValues::try_new(inner.values().as_ref(), function_name)?;
        Ok(Self {
            outer,
            inner,
            values,
        })
    }

    fn is_valid(&self, row: usize) -> bool {
        self.outer.is_valid(row)
    }

    fn row_range(&self, row: usize) -> Range<usize> {
        let start = self.outer.offsets()[row] as usize;
        let len = self.outer.offsets().length_at(row);
        start..start + len
    }

    fn row_len(&self, row: usize) -> usize {
        self.outer.offsets().length_at(row)
    }

    fn list_is_valid(&self, list_index: usize) -> bool {
        self.inner.is_valid(list_index)
    }

    fn document_weights(&self, list_index: usize, p: f64) -> PolarsResult<DocumentWeights> {
        let mut weights = DocumentWeights::new();
        let start = self.inner.offsets()[list_index] as usize;
        let len = self.inner.offsets().length_at(list_index);
        self.values.extend_weights(start, len, p, &mut weights)?;
        Ok(weights)
    }
}

fn self_row_scores(
    lists: &ListCollection<'_>,
    row: usize,
    p: f64,
) -> PolarsResult<Option<Vec<Option<f64>>>> {
    if !lists.is_valid(row) {
        return Ok(None);
    }

    let weights = row_weights(lists, lists.row_range(row), p)?;
    let len = weights.len();
    let mut sums = vec![0.0; len];
    let mut counts = vec![0usize; len];

    for i in 0..len {
        for j in i + 1..len {
            let (Some(left), Some(right)) = (&weights[i], &weights[j]) else {
                continue;
            };

            let score = weighted_jaccard(left, right);
            sums[i] += score;
            sums[j] += score;
            counts[i] += 1;
            counts[j] += 1;
        }
    }

    Ok(Some(mean_scores(weights.iter(), &sums, &counts)))
}

fn cross_row_scores(
    target: &ListCollection<'_>,
    reference: &ListCollection<'_>,
    row: usize,
    p: f64,
) -> PolarsResult<Option<Vec<Option<f64>>>> {
    if !target.is_valid(row) {
        return Ok(None);
    }

    let target_weights = row_weights(target, target.row_range(row), p)?;
    if !reference.is_valid(row) {
        return Ok(Some(vec![None; target_weights.len()]));
    }

    let reference_weights = row_weights(reference, reference.row_range(row), p)?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    if reference_weights.is_empty() {
        return Ok(Some(vec![None; target_weights.len()]));
    }

    let mut out = Vec::with_capacity(target_weights.len());
    for target_weights in target_weights {
        let Some(target_weights) = target_weights else {
            out.push(None);
            continue;
        };

        let sum = reference_weights
            .iter()
            .map(|reference_weights| weighted_jaccard(&target_weights, reference_weights))
            .sum::<f64>();
        out.push(Some(sum / reference_weights.len() as f64));
    }

    Ok(Some(out))
}

fn row_weights(
    lists: &ListCollection<'_>,
    range: Range<usize>,
    p: f64,
) -> PolarsResult<Vec<Option<DocumentWeights>>> {
    range
        .map(|list_index| {
            if lists.list_is_valid(list_index) {
                lists.document_weights(list_index, p).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect()
}

fn mean_scores<'a>(
    weights: impl Iterator<Item = &'a Option<DocumentWeights>>,
    sums: &[f64],
    counts: &[usize],
) -> Vec<Option<f64>> {
    weights
        .enumerate()
        .map(|(index, weights)| {
            if weights.is_some() && counts[index] > 0 {
                Some(sums[index] / counts[index] as f64)
            } else {
                None
            }
        })
        .collect()
}

fn output_builder(
    name: PlSmallStr,
    rows: usize,
    target: &ListCollection<'_>,
) -> ListPrimitiveChunkedBuilder<Float64Type> {
    let values_capacity = (0..rows)
        .filter(|&row| target.is_valid(row))
        .map(|row| target.row_len(row))
        .sum();

    ListPrimitiveChunkedBuilder::<Float64Type>::new(name, rows, values_capacity, DataType::Float64)
}

fn fields_from_series(inputs: &[Series]) -> Vec<Field> {
    inputs
        .iter()
        .map(|series| Field::new(series.name().clone(), series.dtype().clone()))
        .collect()
}

fn validate_self_inputs(input_fields: &[Field]) -> PolarsResult<()> {
    polars_ensure!(
        input_fields.len() == 1,
        ComputeError: "list_mean_similarity expects exactly 1 column, got {}",
        input_fields.len()
    );

    nested_document_dtype(input_fields[0].dtype(), "list_mean_similarity")?;
    Ok(())
}

fn validate_cross_inputs(input_fields: &[Field]) -> PolarsResult<()> {
    polars_ensure!(
        input_fields.len() == 2,
        ComputeError: "list_mean_similarity_to expects exactly 2 columns, got {}",
        input_fields.len()
    );

    let target_dtype = nested_document_dtype(input_fields[0].dtype(), "list_mean_similarity_to")?;
    let reference_dtype =
        nested_document_dtype(input_fields[1].dtype(), "list_mean_similarity_to")?;
    validate_document_dtypes(target_dtype, reference_dtype)
}

fn nested_document_dtype(dtype: &DataType, function_name: &str) -> PolarsResult<DocumentDtype> {
    let inner_list_dtype = list_inner_dtype(dtype)?;
    let document_dtype = list_inner_dtype(inner_list_dtype)?;
    document_dtype_from_dtype(document_dtype, function_name)
}
