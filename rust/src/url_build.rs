use std::borrow::Cow;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn url_build_output(input_fields: &[Field]) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 6,
        ComputeError: "url_build expects 6 component columns, got {}",
        input_fields.len()
    );
    for field in input_fields {
        polars_ensure!(
            field.dtype() == &DataType::String,
            SchemaMismatch:
            "url_build expects String expressions, got `{}` for `{}`",
            field.dtype(),
            field.name()
        );
    }
    Ok(Field::new("url".into(), DataType::String))
}

#[polars_expr(output_type_func=url_build_output)]
fn url_build(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 6,
        ComputeError: "url_build expects 6 component columns, got {}",
        inputs.len()
    );
    let target_len = inputs.iter().map(|series| series.len()).max().unwrap_or(0);
    let mut prepared = Vec::with_capacity(inputs.len());
    for input in inputs {
        let strings = input.str()?;
        polars_ensure!(
            strings.len() == target_len || strings.len() == 1,
            ShapeMismatch:
            "series length {} does not match expected length of {}",
            strings.len(),
            target_len
        );
        let prepared_input = if strings.len() == target_len {
            Cow::Borrowed(strings)
        } else {
            Cow::Owned(strings.new_from_index(0, target_len))
        };
        prepared.push(prepared_input.into_owned());
    }

    let mut urls = Vec::with_capacity(target_len);
    for row in 0..target_len {
        let values = prepared
            .iter()
            .map(|column| column.get(row).unwrap_or(""))
            .collect::<Vec<_>>();
        urls.push(build_url(
            values[0], values[1], values[2], values[3], values[4], values[5],
        ));
    }

    let mut output: StringChunked = urls.into_iter().collect();
    output.rename("url".into());
    Ok(output.into_series())
}

fn build_url(
    scheme: &str,
    netloc: &str,
    path: &str,
    params: &str,
    query: &str,
    fragment: &str,
) -> String {
    let mut path = path.to_string();
    if !params.is_empty() {
        path.push(';');
        path.push_str(params);
    }

    if !netloc.is_empty() || (!scheme.is_empty() && path.starts_with("//")) {
        if !path.is_empty() && !path.starts_with('/') {
            path.insert(0, '/');
        }
        path = format!("//{netloc}{path}");
    }
    if !scheme.is_empty() {
        path = format!("{scheme}:{path}");
    }
    if !query.is_empty() {
        path.push('?');
        path.push_str(query);
    }
    if !fragment.is_empty() {
        path.push('#');
        path.push_str(fragment);
    }
    path
}
