use std::io;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::DomException;

/// Original browser error fields from a permission request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebPermissionError {
    /// Native legacy DOM or geolocation code, when the browser provides one.
    pub code: Option<u32>,
    /// Native exception/error name; geolocation errors have no name member.
    pub name: Option<String>,
    /// Actual native message, or the exact string rejection value.
    pub message: String,
}

pub(super) fn native(value: &JsValue) -> WebPermissionError {
    if let Some(error) = value.dyn_ref::<DomException>() {
        return WebPermissionError {
            code: Some(u32::from(error.code())),
            name: Some(error.name()),
            message: error.message(),
        };
    }
    if let Some(error) = value.dyn_ref::<js_sys::Error>() {
        return WebPermissionError {
            code: None,
            name: Some(error.name().into()),
            message: error.message().into(),
        };
    }
    WebPermissionError {
        code: None,
        name: None,
        message: value.as_string().unwrap_or_else(|| format!("{value:?}")),
    }
}

pub(super) fn interface(
    value: &JsValue,
    name: &str,
) -> Result<Option<JsValue>, WebPermissionError> {
    let value =
        js_sys::Reflect::get(value, &JsValue::from_str(name)).map_err(|value| native(&value))?;
    Ok((!value.is_null() && !value.is_undefined()).then_some(value))
}

pub(super) fn method(value: &JsValue, name: &str) -> Result<bool, WebPermissionError> {
    let Some(value) = interface(value, name)? else {
        return Ok(false);
    };
    if value.is_function() {
        return Ok(true);
    }
    Err(WebPermissionError {
        code: None,
        name: None,
        message: format!(
            "Browser permission member {name} is {}, not callable",
            value
                .js_typeof()
                .as_string()
                .unwrap_or_else(|| format!("{:?}", value.js_typeof()))
        ),
    })
}

pub(super) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
