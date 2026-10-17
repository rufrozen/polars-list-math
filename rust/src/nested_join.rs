use arrow::array::builder::{make_builder, ShareStrategy};
use arrow::array::{
    Array, BinaryViewArray, BooleanArray, PrimitiveArray, StructArray, Utf8ViewArray,
};
use arrow::datatypes::{PhysicalType, PrimitiveType};
use arrow::legacy::prelude::LargeListArray;
use polars_core::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::hash::BuildHasher;

#[derive(Clone, Deserialize)]
struct LookupArgs {
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    value_fields: Vec<String>,
    output_names: Vec<String>,
    skip_null_keys: bool,
}

fn depth(mut dtype: &DataType) -> (usize, &DataType) {
    let mut n = 0;
    while let DataType::List(inner) = dtype {
        n += 1;
        dtype = inner;
    }
    (n, dtype)
}

fn supported(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Boolean
            | DataType::String
            | DataType::Binary
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int128
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn right_fields(dtype: &DataType) -> PolarsResult<&[Field]> {
    if let DataType::List(inner) = dtype {
        if let DataType::Struct(fields) = inner.as_ref() {
            return Ok(fields);
        }
    }
    polars_bail!(SchemaMismatch: "right must have dtype List[Struct]")
}

fn field_index(fields: &[Field], name: &str) -> PolarsResult<usize> {
    fields
        .iter()
        .position(|field| field.name().as_str() == name)
        .ok_or_else(|| polars_err!(ColumnNotFound: "{}", name))
}

fn output_field(fields: &[Field], args: LookupArgs) -> PolarsResult<Field> {
    polars_ensure!(fields.len() == args.left_keys.len() + 2, ComputeError: "invalid lookup inputs");
    let right = right_fields(fields[0].dtype())?;
    let (shape_depth, _) = depth(fields[1].dtype());
    polars_ensure!(shape_depth > 0, SchemaMismatch: "shape must have List dtype");
    for ((field, name), right_name) in fields[2..]
        .iter()
        .zip(&args.left_keys)
        .zip(&args.right_keys)
    {
        let (key_depth, leaf) = depth(field.dtype());
        let right_type = right[field_index(right, right_name)?].dtype();
        polars_ensure!(key_depth <= shape_depth, SchemaMismatch: "key '{}' is deeper than shape", name);
        polars_ensure!(supported(leaf) && supported(right_type), SchemaMismatch: "unsupported key dtype for '{}': {} / {}", name, leaf, right_type);
        polars_ensure!(leaf == right_type, SchemaMismatch: "key dtype mismatch for '{}': {} != {}", name, leaf, right_type);
    }
    let mut output = Vec::with_capacity(args.value_fields.len());
    for (value, name) in args.value_fields.iter().zip(&args.output_names) {
        let mut dtype = right[field_index(right, value)?].dtype().clone();
        for _ in 0..shape_depth {
            dtype = DataType::List(Box::new(dtype));
        }
        output.push(Field::new(name.into(), dtype));
    }
    Ok(Field::new("nested_lookup".into(), DataType::Struct(output)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KeyPart<'a> {
    Integer(i128),
    Boolean(bool),
    String(&'a str),
    Binary(&'a [u8]),
}

fn key_part(array: &dyn Array, index: usize) -> PolarsResult<Option<KeyPart<'_>>> {
    if !array.is_valid(index) {
        return Ok(None);
    }
    macro_rules! integer {
        ($t:ty) => {
            KeyPart::Integer(
                array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$t>>()
                    .unwrap()
                    .value(index) as i128,
            )
        };
    }
    let value = match array.dtype().to_physical_type() {
        PhysicalType::Primitive(PrimitiveType::Int8) => integer!(i8),
        PhysicalType::Primitive(PrimitiveType::Int16) => integer!(i16),
        PhysicalType::Primitive(PrimitiveType::Int32) => integer!(i32),
        PhysicalType::Primitive(PrimitiveType::Int64) => integer!(i64),
        PhysicalType::Primitive(PrimitiveType::Int128) => integer!(i128),
        PhysicalType::Primitive(PrimitiveType::UInt8) => integer!(u8),
        PhysicalType::Primitive(PrimitiveType::UInt16) => integer!(u16),
        PhysicalType::Primitive(PrimitiveType::UInt32) => integer!(u32),
        PhysicalType::Primitive(PrimitiveType::UInt64) => integer!(u64),
        PhysicalType::Boolean => KeyPart::Boolean(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(index),
        ),
        PhysicalType::Utf8View => KeyPart::String(
            array
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .value(index),
        ),
        PhysicalType::BinaryView => KeyPart::Binary(
            array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap()
                .value(index),
        ),
        _ => polars_bail!(SchemaMismatch: "unsupported physical key dtype: {:?}", array.dtype()),
    };
    Ok(Some(value))
}

struct NestedColumn<'a> {
    levels: Vec<&'a LargeListArray>,
    leaf: &'a dyn Array,
}

impl<'a> NestedColumn<'a> {
    fn new(mut array: &'a dyn Array) -> Self {
        let mut levels = Vec::new();
        while let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
            levels.push(list);
            array = list.values().as_ref();
        }
        Self {
            levels,
            leaf: array,
        }
    }
}

struct LookupIndex<'a> {
    // Hashes only select candidates; equality always uses the original columns.
    // Collision links avoid a separate key allocation for every right record.
    rows: PlHashMap<u64, usize>,
    collisions: PlHashMap<usize, usize>,
    keys: Vec<&'a dyn Array>,
}

impl LookupIndex<'_> {
    fn hash(&self, parts: &[KeyPart<'_>]) -> u64 {
        self.rows.hasher().hash_one(parts)
    }

    fn lookup_hash(&self, hash: u64, parts: &[KeyPart<'_>]) -> PolarsResult<Option<usize>> {
        let mut candidate = self.rows.get(&hash).copied();
        while let Some(row) = candidate {
            let mut equal = true;
            for (array, part) in self.keys.iter().zip(parts) {
                if key_part(*array, row)?.as_ref() != Some(part) {
                    equal = false;
                    break;
                }
            }
            if equal {
                return Ok(Some(row));
            }
            candidate = self.collisions.get(&row).copied();
        }
        Ok(None)
    }

    fn lookup(&self, parts: &[KeyPart<'_>]) -> PolarsResult<Option<usize>> {
        self.lookup_hash(self.hash(parts), parts)
    }

    fn insert(
        &mut self,
        hash: u64,
        parts: &[KeyPart<'_>],
        row: usize,
        args: &LookupArgs,
    ) -> PolarsResult<()> {
        if self.lookup_hash(hash, parts)?.is_some() {
            polars_bail!(ComputeError: "duplicate right key: {:?} = {:?}", args.right_keys, parts);
        }
        if let Some(previous) = self.rows.insert(hash, row) {
            self.collisions.insert(row, previous);
        }
        Ok(())
    }
}

fn build_index<'a>(
    list: &LargeListArray,
    records: &'a StructArray,
    key_arrays: &[&'a dyn Array],
    row: usize,
    args: &LookupArgs,
) -> PolarsResult<LookupIndex<'a>> {
    polars_ensure!(list.is_valid(row), ComputeError: "right dictionary must not be null; use an empty list");
    let (start, end) = list.offsets().start_end(row);
    let mut index = LookupIndex {
        rows: PlHashMap::with_capacity(end - start),
        collisions: PlHashMap::new(),
        keys: key_arrays.to_vec(),
    };
    let mut parts = Vec::with_capacity(key_arrays.len());
    for row_index in start..end {
        parts.clear();
        if records.is_valid(row_index) {
            for array in key_arrays {
                if let Some(part) = key_part(*array, row_index)? {
                    parts.push(part);
                } else {
                    break;
                }
            }
        }
        if parts.len() != key_arrays.len() {
            polars_ensure!(args.skip_null_keys, ComputeError: "null in right key at dictionary row {}, record {}", row, row_index - start);
            continue;
        }
        let hash = index.hash(&parts);
        index.insert(hash, &parts, row_index, args)?;
    }
    Ok(index)
}

struct Walker<'a, 'b> {
    shape: &'b NestedColumn<'a>,
    keys: &'b [NestedColumn<'a>],
    args: &'b LookupArgs,
    matches: &'b mut [Option<usize>],
    probe: Vec<KeyPart<'a>>,
    // Recursion uses distinct scratch space per depth. Key positions are
    // physical offsets in immutable left arrays, also across row-local indexes.
    starts: Vec<Vec<Option<usize>>>,
    cached_positions: Vec<usize>,
    cached_parts: Vec<Option<KeyPart<'a>>>,
}

impl<'a> Walker<'a, '_> {
    fn walk(
        &mut self,
        level: usize,
        position: usize,
        key_positions: &mut [usize],
        index: &LookupIndex<'a>,
    ) -> PolarsResult<()> {
        if level == self.shape.levels.len() {
            // Reuse one probe buffer across all left leaves.
            self.probe.clear();
            for (i, key) in self.keys.iter().enumerate() {
                if self.cached_positions[i] != key_positions[i] {
                    self.cached_parts[i] = key_part(key.leaf, key_positions[i])?;
                    self.cached_positions[i] = key_positions[i];
                }
                if let Some(part) = self.cached_parts[i] {
                    self.probe.push(part);
                } else {
                    polars_ensure!(self.args.skip_null_keys, ComputeError: "null in left key '{}' at leaf position {}", self.args.left_keys[i], position);
                    return Ok(());
                }
            }
            self.matches[position] = index.lookup(&self.probe)?;
            return Ok(());
        }
        let shape = self.shape.levels[level];
        if !shape.is_valid(position) {
            return Ok(());
        }
        let (start, end) = shape.offsets().start_end(position);
        for (i, key) in self.keys.iter().enumerate() {
            if let Some(list) = key.levels.get(level) {
                let key_position = key_positions[i];
                polars_ensure!(list.is_valid(key_position) && list.offsets().length_at(key_position) == end - start,
                    ShapeMismatch: "key '{}' does not align with shape at level {}, position {}", self.args.left_keys[i], level, position);
                self.starts[level][i] = Some(list.offsets().start_end(key_position).0);
            } else {
                self.starts[level][i] = None;
            }
        }
        for position in start..end {
            for (i, key_start) in self.starts[level].iter().enumerate() {
                if let Some(key_start) = key_start {
                    key_positions[i] = key_start + position - start;
                }
            }
            self.walk(level + 1, position, key_positions, index)?;
        }
        Ok(())
    }
}

#[polars_expr(output_type_func_with_kwargs=output_field)]
fn nested_lookup(inputs: &[Series], kwargs: LookupArgs) -> PolarsResult<Series> {
    let args = kwargs;
    let fields = inputs
        .iter()
        .map(|s| Field::new(s.name().clone(), s.dtype().clone()))
        .collect::<Vec<_>>();
    let output = output_field(&fields, args.clone())?;
    let DataType::Struct(output_fields) = output.dtype() else {
        unreachable!()
    };
    let prepared = inputs.iter().map(|s| s.rechunk()).collect::<Vec<_>>();
    let arrays = prepared
        .iter()
        .map(|s| s.chunks()[0].as_ref())
        .collect::<Vec<_>>();
    let rows = inputs[1].len();
    polars_ensure!(inputs[0].len() == 1 || inputs[0].len() == rows,
        ShapeMismatch: "lookup inputs must match shape row count or have length 1");
    for key in &inputs[2..] {
        polars_ensure!(key.len() == rows || key.len() == 1,
            ShapeMismatch: "lookup inputs must match shape row count or have length 1");
    }
    let right = arrays[0].as_any().downcast_ref::<LargeListArray>().unwrap();
    let records = right
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let right_schema = right_fields(inputs[0].dtype())?;
    let key_arrays = args
        .right_keys
        .iter()
        .map(|name| Ok(records.values()[field_index(right_schema, name)?].as_ref()))
        .collect::<PolarsResult<Vec<_>>>()?;
    let shape = NestedColumn::new(arrays[1]);
    let keys = arrays[2..]
        .iter()
        .map(|array| NestedColumn::new(*array))
        .collect::<Vec<_>>();
    let mut matches = vec![None; shape.leaf.len()];
    let mut walker = Walker {
        shape: &shape,
        keys: &keys,
        args: &args,
        matches: &mut matches,
        probe: Vec::with_capacity(keys.len()),
        starts: vec![vec![None; keys.len()]; shape.levels.len()],
        cached_positions: vec![usize::MAX; keys.len()],
        cached_parts: vec![None; keys.len()],
    };
    if right.len() == 1 {
        // Validate the complete dictionary, also when rows == 0.
        let index = build_index(right, records, &key_arrays, 0, &args)?;
        for row in 0..rows {
            let mut positions = inputs[2..]
                .iter()
                .map(|s| if s.len() == 1 { 0 } else { row })
                .collect::<Vec<_>>();
            walker.walk(0, row, &mut positions, &index)?;
        }
    } else {
        for row in 0..rows {
            let index = build_index(right, records, &key_arrays, row, &args)?;
            let mut positions = inputs[2..]
                .iter()
                .map(|s| if s.len() == 1 { 0 } else { row })
                .collect::<Vec<_>>();
            walker.walk(0, row, &mut positions, &index)?;
        }
    }
    // IdxSize::MAX is outside the source and gathers a null. Keep the scalar
    // path for dictionaries too large to reserve that sentinel safely.
    let gather_indices = (records.len() <= IdxSize::MAX as usize).then(|| {
        matches
            .iter()
            .map(|index| index.map(|i| i as IdxSize).unwrap_or(IdxSize::MAX))
            .collect::<Vec<_>>()
    });
    let mut output_columns = Vec::with_capacity(args.value_fields.len());
    for (name, output_field) in args.value_fields.iter().zip(output_fields) {
        let source = records.values()[field_index(right_schema, name)?].as_ref();
        let mut builder = make_builder(source.dtype());
        builder.reserve(matches.len());
        if let Some(indices) = &gather_indices {
            builder.opt_gather_extend(source, indices, ShareStrategy::Always);
        } else {
            for index in &matches {
                if let Some(index) = index {
                    builder.subslice_extend(source, *index, 1, ShareStrategy::Always);
                } else {
                    builder.extend_nulls(1);
                }
            }
        }
        let mut array = builder.freeze_reset();
        for list in shape.levels.iter().rev() {
            array = Box::new(LargeListArray::new(
                LargeListArray::default_datatype(array.dtype().clone()),
                list.offsets().clone(),
                array,
                list.validity().cloned(),
            ));
        }
        // Builders preserve the right logical value dtype, and only List wrappers
        // were added. The offsets/validities come directly from the shape arrays.
        let column = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                output_field.name().clone(),
                vec![array],
                output_field.dtype(),
            )
        };
        output_columns.push(column);
    }
    Ok(
        StructChunked::from_series("nested_lookup".into(), rows, output_columns.iter())?
            .into_series(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding_keys_remain_distinct_and_duplicates_are_rejected() -> PolarsResult<()> {
        let numbers = PrimitiveArray::<i64>::from_slice([1, 2, 1, 1]);
        let strings = Utf8ViewArray::from_slice_values(["same", "same", "other", "same"]);
        let mut index = LookupIndex {
            rows: PlHashMap::new(),
            collisions: PlHashMap::new(),
            keys: vec![&numbers, &strings],
        };
        let args = LookupArgs {
            left_keys: vec!["number".into(), "text".into()],
            right_keys: vec!["number".into(), "text".into()],
            value_fields: vec![],
            output_names: vec![],
            skip_null_keys: false,
        };
        let keys = [
            [KeyPart::Integer(1), KeyPart::String("same")],
            [KeyPart::Integer(2), KeyPart::String("same")],
            [KeyPart::Integer(1), KeyPart::String("other")],
        ];
        // Exercise a deterministic collision chain, including a duplicate
        // that matches its oldest entry, without runtime diagnostic settings.
        for (row, key) in keys.iter().enumerate() {
            index.insert(0, key, row, &args)?;
        }
        for (row, key) in keys.iter().enumerate() {
            assert_eq!(index.lookup_hash(0, key)?, Some(row));
        }
        assert_eq!(
            index.lookup_hash(0, &[KeyPart::Integer(2), KeyPart::String("other")])?,
            None
        );
        let error = index.insert(0, &keys[0], 3, &args).unwrap_err();
        assert!(error.to_string().contains("duplicate right key"));
        // A rejected duplicate must not corrupt the existing chain.
        assert_eq!(index.lookup_hash(0, &keys[0])?, Some(0));
        Ok(())
    }
}
