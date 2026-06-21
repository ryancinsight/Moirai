//! PyO3 boundary for `moirai::Moirai`.

#![allow(
    clippy::useless_conversion,
    reason = "PyO3 pymethod expansion emits boundary conversion code that Clippy reports inside the macro"
)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Registers native classes on the Python module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Runtime>()?;
    Ok(())
}

/// PyO3 wrapper around `moirai::Moirai`.
#[pyclass(name = "Runtime")]
struct Runtime {
    inner: moirai::Moirai,
}

#[pymethods]
impl Runtime {
    /// Creates a Moirai runtime with a fixed worker count.
    #[new]
    fn new(workers: usize) -> PyResult<Self> {
        if workers == 0 {
            return Err(PyValueError::new_err("worker count must be at least 1"));
        }

        let inner = moirai::Moirai::builder()
            .worker_threads(workers)
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        Ok(Self { inner })
    }

    /// Returns the configured Moirai worker count.
    fn worker_count(&self) -> usize {
        self.inner.worker_count()
    }

    /// Returns whether queued or active Moirai work exists.
    fn has_work(&self) -> bool {
        self.inner.has_work()
    }

    /// Waits for currently queued and active Moirai work to complete.
    fn join(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .join()
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }

    /// Shuts the wrapped Moirai runtime down.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_moirai_runtime_reports_worker_count() {
        let runtime = moirai::Moirai::builder().worker_threads(2).build().unwrap();
        assert_eq!(runtime.worker_count(), 2);
        assert!(!runtime.has_work());
        runtime.join().unwrap();
        runtime.shutdown();
    }
}
