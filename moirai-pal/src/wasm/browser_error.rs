//! JavaScript exception context at the actual browser boundary.

use std::io;

use wasm_bindgen::{JsCast, JsValue};

pub(crate) fn javascript_error(operation: &str, kind: io::ErrorKind, value: JsValue) -> io::Error {
    let detail = if let Some(error) = value.dyn_ref::<js_sys::Error>() {
        format!(
            "{}: {}",
            String::from(error.name()),
            String::from(error.message())
        )
    } else if let Some(text) = value.dyn_ref::<js_sys::JsString>() {
        String::from(text)
    } else {
        format!("JavaScript threw {value:?}")
    };
    io::Error::new(kind, format!("{operation}: {detail}"))
}
