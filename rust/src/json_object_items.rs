use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
struct JsonObjectItemsKwargs {
    strict: bool,
}

fn json_object_items_output(
    input_fields: &[Field],
    _kwargs: JsonObjectItemsKwargs,
) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 1,
        ComputeError: "json_object_items expects 1 String column, got {}",
        input_fields.len()
    );
    polars_ensure!(
        input_fields[0].dtype() == &DataType::String,
        SchemaMismatch:
        "json_object_items expects a String expression, got `{}`",
        input_fields[0].dtype()
    );

    Ok(Field::new(input_fields[0].name().clone(), output_dtype()))
}

#[polars_expr(output_type_func_with_kwargs=json_object_items_output)]
fn json_object_items(inputs: &[Series], kwargs: JsonObjectItemsKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "json_object_items expects 1 String column, got {}",
        inputs.len()
    );

    let input = inputs[0].str()?;
    let mut rows = Vec::with_capacity(input.len());

    for (row_index, input_value) in input.iter().enumerate() {
        let Some(input_value) = input_value else {
            rows.push(None);
            continue;
        };

        let decoded = match serde_json::from_str::<Value>(input_value) {
            Ok(value) => value,
            Err(error) if kwargs.strict => {
                polars_bail!(
                    ComputeError:
                    "invalid JSON at row {}: {}",
                    row_index,
                    error
                )
            }
            Err(_) => {
                rows.push(None);
                continue;
            }
        };

        let Value::Object(object) = decoded else {
            rows.push(None);
            continue;
        };

        let mut keys = Vec::with_capacity(object.len());
        let mut values = Vec::with_capacity(object.len());
        for (key, value) in object {
            keys.push(key);
            values.push(json_value_to_string(value));
        }

        let mut key_column: StringChunked = keys.into_iter().collect();
        key_column.rename("key".into());
        let mut value_column: StringChunked = values.into_iter().collect();
        value_column.rename("value".into());
        let fields = [key_column.into_series(), value_column.into_series()];
        let row = StructChunked::from_series("".into(), fields[0].len(), fields.iter())?;
        rows.push(Some(row.into_series()));
    }

    let mut output = rows.into_iter().collect::<ListChunked>().into_series();
    output.rename(inputs[0].name().clone());
    output.cast(&output_dtype())
}

fn json_value_to_string(value: Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => Some(value),
        value => Some(value.to_string()),
    }
}

fn output_dtype() -> DataType {
    DataType::List(Box::new(DataType::Struct(vec![
        Field::new("key".into(), DataType::String),
        Field::new("value".into(), DataType::String),
    ])))
}
