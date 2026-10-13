//! [Notifications](https://notifications.spec.whatwg.org/) returns the exact
//! granted/denied/default value; requestPermission has no abort parameter.

use super::{
    WebNotificationPermission, WebPermissionOutcome, error,
    request::{Completion, malformed},
};
use std::rc::Rc;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Notification, Window};

pub(super) fn start(window: &Window, completion: Rc<Completion>) {
    let notification = match error::interface(window.as_ref(), "Notification") {
        Ok(Some(value)) => value,
        Ok(None) => {
            completion.unavailable();
            return;
        }
        Err(error) => {
            completion.reject(error);
            return;
        }
    };
    match error::method(&notification, "requestPermission") {
        Ok(true) => (),
        Ok(false) => {
            completion.unavailable();
            return;
        }
        Err(error) => {
            completion.reject(error);
            return;
        }
    }
    let promise = match Notification::request_permission() {
        Ok(promise) => promise,
        Err(value) => {
            completion.reject(error::native(&value));
            return;
        }
    };
    wasm_bindgen_futures::spawn_local(async move {
        match JsFuture::from(promise).await {
            Ok(value) => {
                let result = match value.as_string().as_deref() {
                    Some("granted") => Ok(WebNotificationPermission::Granted),
                    Some("denied") => Ok(WebNotificationPermission::Denied),
                    Some("default") => Ok(WebNotificationPermission::Default),
                    _ => Err(malformed(
                        "Browser notification permission has an invalid value",
                    )),
                };
                completion.finish(result.map(WebPermissionOutcome::Notification));
            }
            Err(value) => completion.reject(error::native(&value)),
        }
    });
}
