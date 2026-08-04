mod json_array_values;
mod json_object_items;
mod list_combinations;
mod list_mean_similarity;
mod list_similarity;
mod list_similarity_core;
mod list_zip;
mod url_build;
mod url_query_encode;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

// Polars loads expression symbols directly, while maturin expects the shared
// object to also be importable as the internal Python extension module.
#[pymodule]
fn _native(_py: Python<'_>, _module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
