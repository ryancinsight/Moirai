//! Checked methods on the actual native object, without a DOM prototype cast.

use std::io;

use js_sys::{Function, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::wasm::browser_error::javascript_error;

#[derive(Clone)]
pub(super) struct NativeMessageTarget {
    pub(super) receiver: JsValue,
    add: Function,
    remove: Function,
}

#[derive(Clone, Copy)]
pub(super) enum EventOperation {
    Attach,
    Detach,
}

impl NativeMessageTarget {
    pub(super) fn new(receiver: JsValue) -> io::Result<Self> {
        if !receiver.is_object() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WebView2 bridge is not an object",
            ));
        }
        let add = callable(&receiver, "addEventListener")?;
        let remove = callable(&receiver, "removeEventListener")?;
        Ok(Self {
            receiver,
            add,
            remove,
        })
    }

    #[inline]
    pub(super) fn event(&self, operation: EventOperation, callback: &JsValue) -> io::Result<()> {
        let (method, context) = match operation {
            EventOperation::Attach => (&self.add, "WebView2 addEventListener"),
            EventOperation::Detach => (&self.remove, "WebView2 removeEventListener"),
        };
        method
            .call2(&self.receiver, &JsValue::from_str("message"), callback)
            .map(|_| ())
            .map_err(|error| javascript_error(context, io::ErrorKind::Other, error))
    }
}

pub(super) fn callable(object: &JsValue, member: &str) -> io::Result<Function> {
    property(object, member)?
        .dyn_into::<Function>()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("WebView2 {member} is not callable"),
            )
        })
}

pub(super) fn property(object: &JsValue, member: &str) -> io::Result<JsValue> {
    if object.is_null() || object.is_undefined() {
        return Err(unavailable(member));
    }
    let value = Reflect::get(object, &JsValue::from_str(member)).map_err(|error| {
        javascript_error(
            &format!("WebView2 {member} lookup"),
            io::ErrorKind::Other,
            error,
        )
    })?;
    if value.is_null() || value.is_undefined() {
        return Err(unavailable(member));
    }
    Ok(value)
}

pub(super) fn unavailable(member: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!("WebView2 bridge has no {member}"),
    )
}

#[cfg(test)]
mod tests;
