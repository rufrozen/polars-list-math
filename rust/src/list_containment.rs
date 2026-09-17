use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::list_similarity_core::{
    document_dtype, list_inner_dtype, prepare_list_inputs, validate_document_dtypes, DocumentList,
    DocumentWeights, ListCollection,
};

fn validate_inputs(fields: &[Field], count: usize, nested: bool) -> PolarsResult<()> {
    polars_ensure!(fields.len() == count, ComputeError: "containment expects {} columns", count);
    let mut dtypes = Vec::with_capacity(count);
    for field in fields {
        let mut dtype = list_inner_dtype(field.dtype())?;
        if nested {
            dtype = list_inner_dtype(dtype)?;
        }
        dtypes.push(document_dtype(dtype, "containment")?);
    }
    if count == 2 {
        validate_document_dtypes(dtypes[0].clone(), dtypes[1].clone())?;
    }
    Ok(())
}

fn output(fields: &[Field], count: usize, nested: bool) -> PolarsResult<Field> {
    validate_inputs(fields, count, nested)?;
    let dtype = if nested {
        DataType::List(Box::new(DataType::Float64))
    } else {
        DataType::Float64
    };
    Ok(Field::new(fields[0].name().clone(), dtype))
}

fn containment_output(fields: &[Field]) -> PolarsResult<Field> {
    output(fields, 2, false)
}

fn mean_output(fields: &[Field]) -> PolarsResult<Field> {
    output(fields, 1, true)
}

fn mean_to_output(fields: &[Field]) -> PolarsResult<Field> {
    output(fields, 2, true)
}

fn prepare(inputs: &[Series], count: usize, nested: bool) -> PolarsResult<Vec<ListChunked>> {
    let fields = inputs
        .iter()
        .map(|s| Field::new(s.name().clone(), s.dtype().clone()))
        .collect::<Vec<_>>();
    validate_inputs(&fields, count, nested)?;
    prepare_list_inputs(inputs)
}

// Reuse similarity's primitive ID normalization with unit weights. Keys form
// a set: ranks and repeated IDs have no effect on containment.
fn intersection_size(left: &DocumentWeights, right: &DocumentWeights) -> usize {
    let (small, large) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    small.keys().filter(|key| large.contains_key(*key)).count()
}

fn score(common: usize, target_len: usize) -> f64 {
    if target_len == 0 {
        0.0
    } else {
        common as f64 / target_len as f64
    }
}

#[polars_expr(output_type_func=containment_output)]
fn list_containment(inputs: &[Series]) -> PolarsResult<Series> {
    let prepared = prepare(inputs, 2, false)?;
    let left = DocumentList::try_new(&prepared[0], "list_containment")?;
    let right = DocumentList::try_new(&prepared[1], "list_containment")?;
    let mut out = Vec::with_capacity(prepared[0].len());
    for row in 0..prepared[0].len() {
        if !left.is_valid(row) || !right.is_valid(row) {
            out.push(None);
            continue;
        }
        let left = left.document_weights(row, 1.0)?;
        let right = right.document_weights(row, 1.0)?;
        out.push(Some(score(intersection_size(&left, &right), left.len())));
    }
    Ok(Float64Chunked::from_iter_options(inputs[0].name().clone(), out.into_iter()).into_series())
}

fn row_sets(lists: &ListCollection<'_>, row: usize) -> PolarsResult<Vec<Option<DocumentWeights>>> {
    lists
        .row_range(row)
        .map(|index| {
            if lists.list_is_valid(index) {
                lists.document_weights(index, 1.0).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect()
}

fn builder(
    name: PlSmallStr,
    rows: usize,
    target: &ListCollection<'_>,
) -> ListPrimitiveChunkedBuilder<Float64Type> {
    let capacity = (0..rows)
        .filter(|&row| target.is_valid(row))
        .map(|row| target.row_len(row))
        .sum();
    ListPrimitiveChunkedBuilder::<Float64Type>::new(name, rows, capacity, DataType::Float64)
}

#[polars_expr(output_type_func=mean_output)]
fn list_mean_containment(inputs: &[Series]) -> PolarsResult<Series> {
    let prepared = prepare(inputs, 1, true)?;
    let lists = ListCollection::try_new(&prepared[0], "list_mean_containment")?;
    let rows = prepared[0].len();
    let mut out = builder(inputs[0].name().clone(), rows, &lists);
    for row in 0..rows {
        if !lists.is_valid(row) {
            out.append_null();
            continue;
        }
        let sets = row_sets(&lists, row)?;
        let mut sums = vec![0.0; sets.len()];
        let mut counts = vec![0usize; sets.len()];
        for i in 0..sets.len() {
            for j in i + 1..sets.len() {
                if let (Some(left), Some(right)) = (&sets[i], &sets[j]) {
                    let common = intersection_size(left, right);
                    sums[i] += score(common, left.len());
                    sums[j] += score(common, right.len());
                    counts[i] += 1;
                    counts[j] += 1;
                }
            }
        }
        out.append_iter(sums.into_iter().zip(counts).map(|(sum, count)| {
            if count == 0 {
                None
            } else {
                Some(sum / count as f64)
            }
        }));
    }
    Ok(out.finish().into_series())
}

#[polars_expr(output_type_func=mean_to_output)]
fn list_mean_containment_to(inputs: &[Series]) -> PolarsResult<Series> {
    let prepared = prepare(inputs, 2, true)?;
    let target = ListCollection::try_new(&prepared[0], "list_mean_containment_to")?;
    let reference = ListCollection::try_new(&prepared[1], "list_mean_containment_to")?;
    let rows = prepared[0].len();
    let mut out = builder(inputs[0].name().clone(), rows, &target);
    for row in 0..rows {
        if !target.is_valid(row) {
            out.append_null();
            continue;
        }
        let targets = row_sets(&target, row)?;
        let references = if reference.is_valid(row) {
            row_sets(&reference, row)?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        out.append_iter(targets.iter().map(|target| {
            let target = target.as_ref()?;
            if references.is_empty() {
                return None;
            }
            let sum = references
                .iter()
                .map(|reference| score(intersection_size(target, reference), target.len()))
                .sum::<f64>();
            Some(sum / references.len() as f64)
        }));
    }
    Ok(out.finish().into_series())
}
