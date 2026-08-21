//! PyO3 extension module for the Moirai runtime.

#![deny(missing_docs)]
pub mod ffi;

use pyo3::prelude::*;

/// Native extension module loaded as `moirai_python._native`.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register(module)
}
