use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::list_similarity_core::{
    document_dtype, validate_document_dtypes, DocumentList, DocumentRanks, ListCollection,
};

#[derive(Clone, Debug, Deserialize)]
struct KendallTauKwargs {
    normalize: bool,
}

fn list_dtype(dtype: &DataType, depth: usize) -> PolarsResult<DataType> {
    if depth == 0 {
        document_dtype(dtype, "Kendall tau")?;
        return Ok(dtype.clone());
    }
    match dtype {
        DataType::List(inner) | DataType::Array(inner, _) => {
            Ok(DataType::List(Box::new(list_dtype(inner, depth - 1)?)))
        }
        _ => polars_bail!(SchemaMismatch: "Kendall tau expects List or Array levels"),
    }
}

fn validate_inputs(fields: &[Field], count: usize, depth: usize) -> PolarsResult<()> {
    polars_ensure!(fields.len() == count, ComputeError: "Kendall tau expects {} columns", count);
    let mut previous = None;
    for field in fields {
        let normalized = list_dtype(field.dtype(), depth)?;
        let mut leaf = &normalized;
        for _ in 0..depth {
            if let DataType::List(inner) = leaf {
                leaf = inner;
            }
        }
        let dtype = document_dtype(leaf, "Kendall tau")?;
        if let Some(previous) = previous {
            validate_document_dtypes(previous, dtype.clone())?;
        }
        previous = Some(dtype);
    }
    Ok(())
}

fn flat_output(fields: &[Field]) -> PolarsResult<Field> {
    validate_inputs(fields, 2, 1)?;
    Ok(Field::new(fields[0].name().clone(), DataType::Float64))
}

fn self_output(fields: &[Field]) -> PolarsResult<Field> {
    nested_output(fields, 1)
}

fn cross_output(fields: &[Field]) -> PolarsResult<Field> {
    nested_output(fields, 2)
}

fn nested_output(fields: &[Field], count: usize) -> PolarsResult<Field> {
    validate_inputs(fields, count, 2)?;
    Ok(Field::new(
        fields[0].name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

fn prepare_inputs(inputs: &[Series], count: usize, depth: usize) -> PolarsResult<Vec<ListChunked>> {
    let fields = inputs
        .iter()
        .map(|s| s.field().into_owned())
        .collect::<Vec<_>>();
    validate_inputs(&fields, count, depth)?;
    // An empty column with a scalar literal must remain empty.
    let rows = inputs
        .iter()
        .map(|series| series.len())
        .find(|&len| len != 1)
        .unwrap_or(1);
    inputs
        .iter()
        .map(|input| {
            polars_ensure!(input.len() == rows || input.len() == 1,
            ShapeMismatch: "input column lengths must match or be 1");
            let input = input.cast(&list_dtype(input.dtype(), depth)?)?;
            let list = input.list()?;
            let list = if list.len() == rows {
                list.clone()
            } else {
                list.new_from_index(0, rows)
            };
            Ok(list.rechunk().into_owned())
        })
        .collect()
}

#[polars_expr(output_type_func=flat_output)]
fn list_kendall_tau(inputs: &[Series], kwargs: KendallTauKwargs) -> PolarsResult<Series> {
    let prepared = prepare_inputs(inputs, 2, 1)?;
    let left = DocumentList::try_new(&prepared[0], "list_kendall_tau")?;
    let right = DocumentList::try_new(&prepared[1], "list_kendall_tau")?;
    let scores = (0..prepared[0].len()).map(|row| {
        if !left.is_valid(row) || !right.is_valid(row) {
            None
        } else {
            kendall_tau(
                &left.document_ranks(row),
                &right.document_ranks(row),
                kwargs.normalize,
            )
        }
    });
    Ok(Float64Chunked::from_iter_options(inputs[0].name().clone(), scores).into_series())
}

#[polars_expr(output_type_func=self_output)]
fn list_mean_kendall_tau(inputs: &[Series], kwargs: KendallTauKwargs) -> PolarsResult<Series> {
    mean_output(inputs, kwargs.normalize, true)
}

#[polars_expr(output_type_func=cross_output)]
fn list_mean_kendall_tau_to(inputs: &[Series], kwargs: KendallTauKwargs) -> PolarsResult<Series> {
    mean_output(inputs, kwargs.normalize, false)
}

fn mean_output(inputs: &[Series], normalize: bool, self_compare: bool) -> PolarsResult<Series> {
    let prepared = prepare_inputs(inputs, if self_compare { 1 } else { 2 }, 2)?;
    let target = ListCollection::try_new(&prepared[0], "list_mean_kendall_tau")?;
    let reference = if self_compare {
        None
    } else {
        Some(ListCollection::try_new(
            &prepared[1],
            "list_mean_kendall_tau_to",
        )?)
    };
    let rows = prepared[0].len();
    let capacity = (0..rows)
        .filter(|&row| target.is_valid(row))
        .map(|row| target.row_len(row))
        .sum();
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        inputs[0].name().clone(),
        rows,
        capacity,
        DataType::Float64,
    );
    for row in 0..rows {
        if !target.is_valid(row) {
            builder.append_null();
            continue;
        }
        let targets = row_ranks(&target, row);
        let references = reference.as_ref().map(|lists| row_ranks(lists, row));
        let peers = references.as_ref().unwrap_or(&targets);
        let mut sums = vec![0.0; targets.len()];
        let mut counts = vec![0usize; targets.len()];
        for (i, left) in targets.iter().enumerate() {
            let Some(left) = left else {
                continue;
            };
            for (j, right) in peers
                .iter()
                .enumerate()
                .skip(if self_compare { i + 1 } else { 0 })
            {
                let Some(right) = right else {
                    continue;
                };
                let Some(score) = kendall_tau(left, right, normalize) else {
                    continue;
                };
                sums[i] += score;
                counts[i] += 1;
                if self_compare {
                    sums[j] += score;
                    counts[j] += 1;
                }
            }
        }
        builder.append_iter(sums.into_iter().zip(counts).map(|(sum, count)| {
            if count == 0 {
                None
            } else {
                Some(sum / count as f64)
            }
        }));
    }
    Ok(builder.finish().into_series())
}

fn row_ranks(lists: &ListCollection<'_>, row: usize) -> Vec<Option<DocumentRanks>> {
    if !lists.is_valid(row) {
        return Vec::new();
    }
    lists
        .row_range(row)
        .map(|index| {
            lists
                .list_is_valid(index)
                .then(|| lists.document_ranks(index))
        })
        .collect()
}

fn kendall_tau(left: &DocumentRanks, right: &DocumentRanks, normalize: bool) -> Option<f64> {
    let mut common = left
        .iter()
        .filter_map(|(key, &rank)| right.get(key).map(|&other| (rank, other)))
        .collect::<Vec<_>>();
    if common.len() < 2 {
        return None;
    }
    common.sort_unstable_by_key(|&(rank, _)| rank);
    let mut tree = vec![0usize; right.len() + 1];
    let mut discordant = 0u64;
    for (seen, &(_, position)) in common.iter().enumerate() {
        let mut index = position + 1;
        let mut prefix = 0;
        while index > 0 {
            prefix += tree[index];
            index &= index - 1;
        }
        discordant += (seen - prefix) as u64;
        index = position + 1;
        while index < tree.len() {
            tree[index] += 1;
            index += index & index.wrapping_neg();
        }
    }
    let count = common.len() as f64;
    let pairs = count * (count - 1.0) / 2.0;
    let tau = (1.0 - 2.0 * discordant as f64 / pairs).clamp(-1.0, 1.0);
    Some(if normalize { (tau + 1.0) / 2.0 } else { tau })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::list_similarity_core::DocumentKey;

    fn ranks(values: &[i128]) -> DocumentRanks {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| (DocumentKey::Signed(v), i))
            .collect()
    }

    #[test]
    fn scores_and_undefined_pairs() {
        let left = ranks(&[1, 2, 3]);
        assert_eq!(kendall_tau(&left, &left, false), Some(1.0));
        assert_eq!(kendall_tau(&left, &ranks(&[3, 2, 1]), false), Some(-1.0));
        assert_eq!(kendall_tau(&left, &ranks(&[3, 2, 1]), true), Some(0.0));
        assert_eq!(kendall_tau(&left, &ranks(&[4, 2, 5]), false), None);
        assert_eq!(kendall_tau(&left, &ranks(&[]), false), None);
    }

    #[test]
    fn inversion_count_does_not_overflow_u32() {
        let values = (0..100_000).collect::<Vec<i128>>();
        let reversed = values.iter().rev().copied().collect::<Vec<_>>();
        assert_eq!(
            kendall_tau(&ranks(&values), &ranks(&reversed), false),
            Some(-1.0)
        );
    }
}
