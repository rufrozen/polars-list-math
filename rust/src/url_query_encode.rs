use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
struct UrlQueryEncodeKwargs {
    doseq: bool,
}

fn url_query_encode_output(
    input_fields: &[Field],
    _kwargs: UrlQueryEncodeKwargs,
) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 1,
        ComputeError: "url_query_encode expects 1 column, got {}",
        input_fields.len()
    );
    match input_fields[0].dtype() {
        DataType::Struct(_) => {}
        DataType::List(inner) if matches!(inner.as_ref(), DataType::Struct(_)) => {}
        dtype => polars_bail!(
            SchemaMismatch:
            "url_query_encode expects a Struct or List(Struct) expression, got `{}`",
            dtype
        ),
    }
    Ok(Field::new(input_fields[0].name().clone(), DataType::String))
}

#[polars_expr(output_type_func_with_kwargs=url_query_encode_output)]
fn url_query_encode(inputs: &[Series], kwargs: UrlQueryEncodeKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "url_query_encode expects 1 column, got {}",
        inputs.len()
    );

    let values = match inputs[0].dtype() {
        DataType::Struct(_) => encode_struct_rows(&inputs[0], kwargs.doseq)?,
        DataType::List(_) => encode_pair_rows(&inputs[0])?,
        dtype => polars_bail!(
            SchemaMismatch:
            "url_query_encode expects a Struct or List(Struct) expression, got `{}`",
            dtype
        ),
    };
    let mut output: StringChunked = values.into_iter().collect();
    output.rename(inputs[0].name().clone());
    Ok(output.into_series())
}

fn encode_struct_rows(input: &Series, doseq: bool) -> PolarsResult<Vec<Option<String>>> {
    let structs = input.struct_()?;
    let fields = structs.fields_as_series();
    let mut rows = Vec::with_capacity(input.len());

    for row in 0..input.len() {
        if input.get(row)?.is_null() {
            rows.push(None);
            continue;
        }
        let mut pairs = Vec::new();
        for field in &fields {
            push_value_pairs(&mut pairs, field.name().as_str(), field.get(row)?, doseq)?;
        }
        rows.push(Some(join_pairs(pairs)));
    }
    Ok(rows)
}

fn encode_pair_rows(input: &Series) -> PolarsResult<Vec<Option<String>>> {
    let lists = input.list()?;
    let mut rows = Vec::with_capacity(input.len());
    for row in 0..input.len() {
        let Some(values) = lists.get_as_series(row) else {
            rows.push(None);
            continue;
        };
        let structs = values.struct_()?;
        let fields = structs.fields_as_series();
        let key = fields.iter().find(|field| field.name().as_str() == "key");
        let value = fields.iter().find(|field| field.name().as_str() == "value");
        let (Some(keys), Some(values)) = (key, value) else {
            polars_bail!(
                SchemaMismatch:
                "url_query_encode expects list structs with `key` and `value` fields"
            )
        };
        polars_ensure!(
            keys.dtype() == &DataType::String,
            SchemaMismatch: "url_query_encode expects `key` to be String"
        );

        let mut pairs = Vec::with_capacity(values.len());
        for index in 0..values.len() {
            let key = keys.get(index)?;
            let Some(key) = key.extract_str() else {
                polars_bail!(ComputeError: "url_query_encode does not allow null keys")
            };
            pairs.push((key.to_string(), scalar_to_string(values.get(index)?)));
        }
        rows.push(Some(join_pairs(pairs)));
    }
    Ok(rows)
}

fn push_value_pairs(
    pairs: &mut Vec<(String, String)>,
    key: &str,
    value: AnyValue<'_>,
    doseq: bool,
) -> PolarsResult<()> {
    if doseq {
        if let AnyValue::List(values) = value {
            for index in 0..values.len() {
                pairs.push((key.to_string(), scalar_to_string(values.get(index)?)));
            }
            return Ok(());
        }
    }
    pairs.push((key.to_string(), value_to_string(value)?));
    Ok(())
}

fn value_to_string(value: AnyValue<'_>) -> PolarsResult<String> {
    match value {
        AnyValue::List(values) => {
            let mut items = Vec::with_capacity(values.len());
            for index in 0..values.len() {
                items.push(python_repr(values.get(index)?));
            }
            Ok(format!("[{}]", items.join(", ")))
        }
        value => Ok(scalar_to_string(value)),
    }
}

fn scalar_to_string(value: AnyValue<'_>) -> String {
    match value {
        AnyValue::Null => "None".to_string(),
        AnyValue::Boolean(true) => "True".to_string(),
        AnyValue::Boolean(false) => "False".to_string(),
        AnyValue::String(value) => value.to_string(),
        AnyValue::StringOwned(value) => value.to_string(),
        value => value.to_string(),
    }
}

fn python_repr(value: AnyValue<'_>) -> String {
    match value {
        AnyValue::String(value) => format!("'{value}'"),
        AnyValue::StringOwned(value) => format!("'{value}'"),
        value => scalar_to_string(value),
    }
}

fn join_pairs(pairs: Vec<(String, String)>) -> String {
    pairs
        .into_iter()
        .map(|(key, value)| format!("{}={}", quote_plus(&key), quote_plus(&value)))
        .collect::<Vec<_>>()
        .join("&")
}

fn quote_plus(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                output.push(char::from(byte));
            }
            b' ' => output.push('+'),
            byte => {
                output.push('%');
                output.push(char::from(HEX[usize::from(byte >> 4)]));
                output.push(char::from(HEX[usize::from(byte & 0x0f)]));
            }
        }
    }
    output
}
