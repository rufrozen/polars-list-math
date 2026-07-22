use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::list_similarity_core::{
    document_dtype, list_inner_dtype, prepare_list_inputs, validate_document_dtypes, validate_p,
    weighted_jaccard, DocumentList,
};

#[derive(Clone, Debug, Deserialize)]
struct ListSimilarityKwargs {
    p: f64,
}

fn list_similarity_output(
    input_fields: &[Field],
    kwargs: ListSimilarityKwargs,
) -> PolarsResult<Field> {
    validate_p(kwargs.p)?;
    validate_inputs(input_fields)?;

    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

#[polars_expr(output_type_func_with_kwargs=list_similarity_output)]
fn list_similarity(inputs: &[Series], kwargs: ListSimilarityKwargs) -> PolarsResult<Series> {
    validate_p(kwargs.p)?;
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "list_similarity expects exactly 2 columns, got {}",
        inputs.len()
    );

    let input_fields: Vec<Field> = inputs
        .iter()
        .map(|series| Field::new(series.name().clone(), series.dtype().clone()))
        .collect();
    validate_inputs(&input_fields)?;

    let prepared = prepare_list_inputs(inputs)?;
    let list_a = DocumentList::try_new(&prepared[0], "list_similarity")?;
    let list_b = DocumentList::try_new(&prepared[1], "list_similarity")?;
    let rows = prepared[0].len();

    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        out.push(list_row_similarity(&list_a, &list_b, row, kwargs.p)?);
    }

    Ok(Float64Chunked::from_iter_options(inputs[0].name().clone(), out.into_iter()).into_series())
}

fn validate_inputs(input_fields: &[Field]) -> PolarsResult<()> {
    polars_ensure!(
        input_fields.len() == 2,
        ComputeError: "list_similarity expects exactly 2 columns, got {}",
        input_fields.len()
    );

    let dtype_a = document_dtype(
        list_inner_dtype(input_fields[0].dtype())?,
        "list_similarity",
    )?;
    let dtype_b = document_dtype(
        list_inner_dtype(input_fields[1].dtype())?,
        "list_similarity",
    )?;
    validate_document_dtypes(dtype_a, dtype_b)
}

fn list_row_similarity(
    list_a: &DocumentList<'_>,
    list_b: &DocumentList<'_>,
    row: usize,
    p: f64,
) -> PolarsResult<Option<f64>> {
    if !list_a.is_valid(row) || !list_b.is_valid(row) {
        return Ok(None);
    }

    let weights_a = list_a.document_weights(row, p)?;
    let weights_b = list_b.document_weights(row, p)?;
    Ok(Some(weighted_jaccard(&weights_a, &weights_b)))
}
