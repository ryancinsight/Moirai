//! PyO3 extension module for the Moirai runtime.

pub mod ffi;

use pyo3::prelude::*;

/// Native extension module loaded as `moirai_python._native`.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register(module)
}
