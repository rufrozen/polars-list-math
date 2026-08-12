use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn expected_value_of_game_output(input_fields: &[Field]) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 1,
        ComputeError: "expected_value_of_game expects exactly 1 column, got {}",
        input_fields.len()
    );
    validate_game_dtype(input_fields[0].dtype())?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

#[polars_expr(output_type_func=expected_value_of_game_output)]
fn expected_value_of_game(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 1,
        ComputeError: "expected_value_of_game expects exactly 1 column, got {}",
        inputs.len()
    );
    validate_game_dtype(inputs[0].dtype())?;
    let games = inputs[0].list()?;
    let mut output = Vec::with_capacity(games.len());
    for row in 0..games.len() {
        let game = games.get_as_series(row).ok_or_else(
            || polars_err!(ComputeError: "expected_value_of_game does not accept null games"),
        )?;
        let states = game.list()?;
        let n = states.len();
        if n < 2 {
            output.push(0.0);
            continue;
        }
        let final_state = states.get_as_series(n - 1).ok_or_else(
            || polars_err!(ComputeError: "expected_value_of_game does not accept null states"),
        )?;
        polars_ensure!(
            final_state.is_empty(),
            ComputeError: "the final state must have no actions"
        );
        let mut evog = vec![0.0; n];
        for i in (0..n - 1).rev() {
            let actions = states.get_as_series(i).ok_or_else(
                || polars_err!(ComputeError: "expected_value_of_game does not accept null states"),
            )?;
            let fields = actions.struct_()?.fields_as_series();
            let probability = field(&fields, "probability")?.cast(&DataType::Float64)?;
            let value = field(&fields, "value")?.cast(&DataType::Float64)?;
            let next_state = field(&fields, "next_state")?.cast(&DataType::Int64)?;
            let probability = probability.f64()?;
            let value = value.f64()?;
            let next_state = next_state.i64()?;
            let mut evos = 0.0;
            for action in 0..actions.len() {
                let prob = probability.get(action).ok_or_else(
                    || polars_err!(ComputeError: "action probability must not be null"),
                )?;
                let action_value = value
                    .get(action)
                    .ok_or_else(|| polars_err!(ComputeError: "action value must not be null"))?;
                let next = next_state.get(action).ok_or_else(
                    || polars_err!(ComputeError: "action next_state must not be null"),
                )?;
                let next = usize::try_from(next).map_err(|_| {
                    polars_err!(ComputeError: "next_state in state {i} must be greater than {i} and less than {n}")
                })?;
                polars_ensure!(
                    next > i && next < n,
                    ComputeError: "next_state in state {i} must be greater than {i} and less than {n}"
                );
                evos += prob * (action_value + evog[next]);
            }
            evog[i] = evos;
        }
        output.push(evog[0]);
    }
    Ok(Float64Chunked::from_vec(inputs[0].name().clone(), output).into_series())
}

fn field<'a>(fields: &'a [Series], name: &str) -> PolarsResult<&'a Series> {
    fields
        .iter()
        .find(|field| field.name().as_str() == name)
        .ok_or_else(|| polars_err!(SchemaMismatch: "action structs must contain `{name}`"))
}

fn validate_game_dtype(dtype: &DataType) -> PolarsResult<()> {
    let DataType::List(states) = dtype else {
        polars_bail!(SchemaMismatch: "expected_value_of_game expects List(List(Struct)), got `{dtype}`")
    };
    let DataType::List(actions) = states.as_ref() else {
        polars_bail!(SchemaMismatch: "expected_value_of_game expects List(List(Struct)), got `{dtype}`")
    };
    let DataType::Struct(fields) = actions.as_ref() else {
        polars_bail!(SchemaMismatch: "expected_value_of_game expects List(List(Struct)), got `{dtype}`")
    };
    for name in ["probability", "value", "next_state"] {
        polars_ensure!(
            fields.iter().any(|field| field.name().as_str() == name),
            SchemaMismatch: "action structs must contain `{name}`"
        );
    }
    Ok(())
}
