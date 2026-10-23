use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, Float16Array, NullArray, PrimitiveArray,
    Utf8Array, Utf8ViewArray,
};
use arrow::legacy::prelude::LargeListArray;
use arrow::offset::Offset;
use arrow::types::NativeType;
use polars::prelude::*;

pub(crate) type DocumentWeights = HashMap<DocumentKey, f64>;
pub(crate) type DocumentRanks = HashMap<DocumentKey, usize>;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum DocumentKey {
    Boolean(bool),
    Signed(i128),
    Unsigned(u128),
    Float(u64),
    String(String),
    Binary(Vec<u8>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DocumentDtype {
    Value(DataType),
    Null,
}

pub(crate) struct DocumentList<'a> {
    list: &'a LargeListArray,
    values: DocumentValues<'a>,
}

pub(crate) enum DocumentValues<'a> {
    Boolean(&'a BooleanArray),
    Int8(&'a PrimitiveArray<i8>),
    Int16(&'a PrimitiveArray<i16>),
    Int32(&'a PrimitiveArray<i32>),
    Int64(&'a PrimitiveArray<i64>),
    Int128(&'a PrimitiveArray<i128>),
    UInt8(&'a PrimitiveArray<u8>),
    UInt16(&'a PrimitiveArray<u16>),
    UInt32(&'a PrimitiveArray<u32>),
    UInt64(&'a PrimitiveArray<u64>),
    UInt128(&'a PrimitiveArray<u128>),
    Float16(&'a Float16Array),
    Float32(&'a PrimitiveArray<f32>),
    Float64(&'a PrimitiveArray<f64>),
    Utf8View(&'a Utf8ViewArray),
    Utf8(&'a Utf8Array<i32>),
    LargeUtf8(&'a Utf8Array<i64>),
    BinaryView(&'a BinaryViewArray),
    Binary(&'a BinaryArray<i32>),
    LargeBinary(&'a BinaryArray<i64>),
    Null,
}

impl<'a> DocumentList<'a> {
    pub(crate) fn try_new(ca: &'a ListChunked, function_name: &str) -> PolarsResult<Self> {
        let list = ca.downcast_as_array();
        let values = DocumentValues::try_new(list.values().as_ref(), function_name)?;
        Ok(Self { list, values })
    }

    pub(crate) fn is_valid(&self, row: usize) -> bool {
        self.list.is_valid(row)
    }

    pub(crate) fn document_weights(&self, row: usize, p: f64) -> PolarsResult<DocumentWeights> {
        let mut weights = DocumentWeights::new();
        let start = self.list.offsets()[row] as usize;
        let len = self.list.offsets().length_at(row);
        self.values.extend_weights(start, len, p, &mut weights)?;
        Ok(weights)
    }

    pub(crate) fn document_ranks(&self, row: usize) -> DocumentRanks {
        self.values.document_ranks(
            self.list.offsets()[row] as usize,
            self.list.offsets().length_at(row),
        )
    }
}

impl<'a> DocumentValues<'a> {
    fn document_ranks(&self, start: usize, len: usize) -> DocumentRanks {
        let mut ranks = DocumentRanks::new();
        macro_rules! read_ranks {
            ($values:expr, $key:expr) => {{
                let values = $values;
                for index in start..start + len {
                    if values.is_valid(index) {
                        let rank = ranks.len();
                        ranks.entry(($key)(values.value(index))).or_insert(rank);
                    }
                }
            }};
        }
        match self {
            Self::Boolean(v) => read_ranks!(v, DocumentKey::Boolean),
            Self::Int8(v) => read_ranks!(v, |x: i8| DocumentKey::Signed(x.into())),
            Self::Int16(v) => read_ranks!(v, |x: i16| DocumentKey::Signed(x.into())),
            Self::Int32(v) => read_ranks!(v, |x: i32| DocumentKey::Signed(x.into())),
            Self::Int64(v) => read_ranks!(v, |x: i64| DocumentKey::Signed(x.into())),
            Self::Int128(v) => read_ranks!(v, DocumentKey::Signed),
            Self::UInt8(v) => read_ranks!(v, |x: u8| DocumentKey::Unsigned(x.into())),
            Self::UInt16(v) => read_ranks!(v, |x: u16| DocumentKey::Unsigned(x.into())),
            Self::UInt32(v) => read_ranks!(v, |x: u32| DocumentKey::Unsigned(x.into())),
            Self::UInt64(v) => read_ranks!(v, |x: u64| DocumentKey::Unsigned(x.into())),
            Self::UInt128(v) => read_ranks!(v, DocumentKey::Unsigned),
            Self::Float16(values) => {
                for index in start..start + len {
                    if values.is_valid(index) {
                        let rank = ranks.len();
                        ranks
                            .entry(DocumentKey::Float(values.value(index).to_bits() as u64))
                            .or_insert(rank);
                    }
                }
            }
            Self::Float32(v) => read_ranks!(v, |x: f32| DocumentKey::Float(x.to_bits() as u64)),
            Self::Float64(v) => read_ranks!(v, |x: f64| DocumentKey::Float(x.to_bits())),
            Self::Utf8View(v) => read_ranks!(v, |x: &str| DocumentKey::String(x.to_owned())),
            Self::Utf8(v) => read_ranks!(v, |x: &str| DocumentKey::String(x.to_owned())),
            Self::LargeUtf8(v) => read_ranks!(v, |x: &str| DocumentKey::String(x.to_owned())),
            Self::BinaryView(v) => read_ranks!(v, |x: &[u8]| DocumentKey::Binary(x.to_vec())),
            Self::Binary(v) => read_ranks!(v, |x: &[u8]| DocumentKey::Binary(x.to_vec())),
            Self::LargeBinary(v) => read_ranks!(v, |x: &[u8]| DocumentKey::Binary(x.to_vec())),
            Self::Null => {}
        }
        ranks
    }

    pub(crate) fn try_new(values: &'a dyn Array, function_name: &str) -> PolarsResult<Self> {
        if let Some(values) = values.as_any().downcast_ref::<BooleanArray>() {
            return Ok(Self::Boolean(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<i8>>() {
            return Ok(Self::Int8(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<i16>>() {
            return Ok(Self::Int16(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<i32>>() {
            return Ok(Self::Int32(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<i64>>() {
            return Ok(Self::Int64(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<i128>>() {
            return Ok(Self::Int128(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<u8>>() {
            return Ok(Self::UInt8(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<u16>>() {
            return Ok(Self::UInt16(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<u32>>() {
            return Ok(Self::UInt32(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<u64>>() {
            return Ok(Self::UInt64(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<u128>>() {
            return Ok(Self::UInt128(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<Float16Array>() {
            return Ok(Self::Float16(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<f32>>() {
            return Ok(Self::Float32(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<PrimitiveArray<f64>>() {
            return Ok(Self::Float64(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<Utf8ViewArray>() {
            return Ok(Self::Utf8View(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<Utf8Array<i32>>() {
            return Ok(Self::Utf8(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<Utf8Array<i64>>() {
            return Ok(Self::LargeUtf8(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<BinaryViewArray>() {
            return Ok(Self::BinaryView(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<BinaryArray<i32>>() {
            return Ok(Self::Binary(values));
        }
        if let Some(values) = values.as_any().downcast_ref::<BinaryArray<i64>>() {
            return Ok(Self::LargeBinary(values));
        }
        if values.as_any().downcast_ref::<NullArray>().is_some() {
            return Ok(Self::Null);
        }

        polars_bail!(
            SchemaMismatch:
            "{} expects a primitive list inner dtype, got Arrow dtype {:?}",
            function_name,
            values.dtype()
        );
    }

    pub(crate) fn extend_weights(
        &self,
        start: usize,
        len: usize,
        p: f64,
        weights: &mut DocumentWeights,
    ) -> PolarsResult<()> {
        match self {
            Self::Boolean(values) => extend_boolean_weights(values, start, len, p, weights),
            Self::Int8(values) => extend_signed_weights(values, start, len, p, weights),
            Self::Int16(values) => extend_signed_weights(values, start, len, p, weights),
            Self::Int32(values) => extend_signed_weights(values, start, len, p, weights),
            Self::Int64(values) => extend_signed_weights(values, start, len, p, weights),
            Self::Int128(values) => extend_signed_weights(values, start, len, p, weights),
            Self::UInt8(values) => extend_unsigned_weights(values, start, len, p, weights),
            Self::UInt16(values) => extend_unsigned_weights(values, start, len, p, weights),
            Self::UInt32(values) => extend_unsigned_weights(values, start, len, p, weights),
            Self::UInt64(values) => extend_unsigned_weights(values, start, len, p, weights),
            Self::UInt128(values) => extend_unsigned_weights(values, start, len, p, weights),
            Self::Float16(values) => extend_float16_weights(values, start, len, p, weights),
            Self::Float32(values) => extend_float32_weights(values, start, len, p, weights),
            Self::Float64(values) => extend_float64_weights(values, start, len, p, weights),
            Self::Utf8View(values) => extend_utf8_view_weights(values, start, len, p, weights),
            Self::Utf8(values) => extend_utf8_weights(values, start, len, p, weights),
            Self::LargeUtf8(values) => extend_utf8_weights(values, start, len, p, weights),
            Self::BinaryView(values) => extend_binary_view_weights(values, start, len, p, weights),
            Self::Binary(values) => extend_binary_weights(values, start, len, p, weights),
            Self::LargeBinary(values) => extend_binary_weights(values, start, len, p, weights),
            Self::Null => Ok(()),
        }
    }
}

pub(crate) fn validate_p(p: f64) -> PolarsResult<()> {
    polars_ensure!(
        p.is_finite() && p > 0.0 && p <= 1.0,
        ComputeError: "p must satisfy 0 < p <= 1"
    );
    Ok(())
}

pub(crate) fn list_inner_dtype(dtype: &DataType) -> PolarsResult<&DataType> {
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

pub(crate) fn document_dtype(dtype: &DataType, function_name: &str) -> PolarsResult<DocumentDtype> {
    if dtype == &DataType::Null {
        return Ok(DocumentDtype::Null);
    }

    if is_supported_document_dtype(dtype) {
        return Ok(DocumentDtype::Value(dtype.clone()));
    }

    polars_bail!(
        SchemaMismatch:
        "{} expects a primitive list inner dtype, got `{}`",
        function_name,
        dtype
    );
}

pub(crate) fn validate_document_dtypes(
    left: DocumentDtype,
    right: DocumentDtype,
) -> PolarsResult<()> {
    if matches!(
        (&left, &right),
        (DocumentDtype::Null, _) | (_, DocumentDtype::Null)
    ) {
        return Ok(());
    }

    if left != right {
        polars_bail!(
            SchemaMismatch:
            "document_id list inner dtypes must match exactly or be Null"
        );
    }
    Ok(())
}

fn is_supported_document_dtype(dtype: &DataType) -> bool {
    dtype.is_primitive() || dtype.is_temporal() || matches!(dtype, DataType::BinaryOffset)
}

pub(crate) fn prepare_list_inputs(inputs: &[Series]) -> PolarsResult<Vec<ListChunked>> {
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

pub(crate) fn weighted_jaccard(weights_a: &DocumentWeights, weights_b: &DocumentWeights) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (document_id, weight_a) in weights_a {
        let weight_b = weights_b.get(document_id).copied().unwrap_or(0.0);
        numerator += weight_a.min(weight_b);
        denominator += weight_a.max(weight_b);
    }

    for (document_id, weight_b) in weights_b {
        if !weights_a.contains_key(document_id) {
            denominator += weight_b;
        }
    }

    if denominator == 0.0 {
        1.0
    } else {
        numerator / denominator
    }
}

fn extend_boolean_weights(
    values: &BooleanArray,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Boolean(values.value(index)))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_signed_weights<T>(
    values: &PrimitiveArray<T>,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()>
where
    T: NativeType + Into<i128>,
{
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Signed(values.value(index).into()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_unsigned_weights<T>(
    values: &PrimitiveArray<T>,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()>
where
    T: NativeType + Into<u128>,
{
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Unsigned(values.value(index).into()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_float16_weights(
    values: &Float16Array,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Float(values.value(index).to_bits() as u64))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_float32_weights(
    values: &PrimitiveArray<f32>,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Float(values.value(index).to_bits() as u64))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_float64_weights(
    values: &PrimitiveArray<f64>,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Float(values.value(index).to_bits()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_utf8_view_weights(
    values: &Utf8ViewArray,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::String(values.value(index).to_string()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_binary_view_weights(
    values: &BinaryViewArray,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Binary(values.value(index).to_vec()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_binary_weights<O: Offset>(
    values: &BinaryArray<O>,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::Binary(values.value(index).to_vec()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

fn extend_utf8_weights<O: Offset>(
    values: &Utf8Array<O>,
    start: usize,
    len: usize,
    p: f64,
    weights: &mut DocumentWeights,
) -> PolarsResult<()> {
    for index in start..start + len {
        if values.is_valid(index) {
            let rank = index - start;
            weights
                .entry(DocumentKey::String(values.value(index).to_string()))
                .or_insert_with(|| p.powf(rank as f64));
        }
    }
    Ok(())
}

pub(crate) struct ListCollection<'a> {
    outer: &'a LargeListArray,
    inner: &'a LargeListArray,
    values: DocumentValues<'a>,
}

impl<'a> ListCollection<'a> {
    pub(crate) fn try_new(ca: &'a ListChunked, function_name: &str) -> PolarsResult<Self> {
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

    pub(crate) fn is_valid(&self, row: usize) -> bool {
        self.outer.is_valid(row)
    }

    pub(crate) fn row_range(&self, row: usize) -> Range<usize> {
        let start = self.outer.offsets()[row] as usize;
        let len = self.outer.offsets().length_at(row);
        start..start + len
    }

    pub(crate) fn row_len(&self, row: usize) -> usize {
        self.outer.offsets().length_at(row)
    }

    pub(crate) fn list_is_valid(&self, list_index: usize) -> bool {
        self.inner.is_valid(list_index)
    }

    pub(crate) fn document_ranks(&self, list_index: usize) -> DocumentRanks {
        self.values.document_ranks(
            self.inner.offsets()[list_index] as usize,
            self.inner.offsets().length_at(list_index),
        )
    }

    pub(crate) fn document_weights(
        &self,
        list_index: usize,
        p: f64,
    ) -> PolarsResult<DocumentWeights> {
        let mut weights = DocumentWeights::new();
        let start = self.inner.offsets()[list_index] as usize;
        let len = self.inner.offsets().length_at(list_index);
        self.values.extend_weights(start, len, p, &mut weights)?;
        Ok(weights)
    }
}
