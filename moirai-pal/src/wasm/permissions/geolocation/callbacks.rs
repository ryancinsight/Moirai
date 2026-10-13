use super::{CallbackKey, Completion, Registration};
use crate::{
    permission_state::callback_key::CallbackSequence,
    wasm::permissions::{WebPermissionError, WebPermissionOutcome, request::malformed},
};
use js_sys::Function;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::{Rc, Weak},
};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{Position, PositionError};

struct Pending {
    registration: Rc<Registration>,
    completion: Weak<Completion>,
}

struct Dispatchers {
    // Condition (1): the browser dynamically invokes these WebIDL callbacks.
    position: Closure<dyn FnMut(JsValue, Position)>,
    // Condition (1): the native callback boundary carries geolocation errors.
    rejected: Closure<dyn FnMut(JsValue, PositionError)>,
}

thread_local! {
    // The realm owns these two entry points even when a document has no watches.
    // Binding a key creates a native JS function, not another Rust closure.
    static DISPATCHERS: RefCell<Option<Dispatchers>> = const { RefCell::new(None) };
    static SEQUENCE: RefCell<CallbackSequence> = RefCell::new(CallbackSequence::default());
    // An unresolved record retains no application DOM or request consumer.
    // The document's occupied Geolocation slot bounds this map across remount.
    static PENDING: RefCell<BTreeMap<CallbackKey, Pending>> = const { RefCell::new(BTreeMap::new()) };
}

fn take(key: &JsValue) -> Option<Pending> {
    let key = key.as_string().and_then(|key| CallbackKey::decode(&key))?;
    PENDING.with(|pending| pending.borrow_mut().remove(&key))
}

fn settle(pending: Pending, result: std::io::Result<WebPermissionOutcome>) {
    pending.registration.clear();
    if let Some(completion) = pending.completion.upgrade() {
        completion.finish(result);
    } else {
        // Consumer cancellation leaves its token DetachedPending. A real late
        // callback releases that guard without resurrecting a request/DOM owner.
        pending
            .registration
            .registry
            .state
            .borrow_mut()
            .settle(pending.registration.token);
    }
}

impl Dispatchers {
    fn new() -> Self {
        // Position/PositionError are stable web-sys bindings for the actual
        // native callback payloads; they require no unstable API configuration.
        // Condition (1): wasm-bindgen erases this browser-selected callback type.
        let position =
            Closure::<dyn FnMut(JsValue, Position)>::new(|key: JsValue, position: Position| {
                let Some(pending) = take(&key) else {
                    return;
                };
                // Inspect native metadata while retaining and publishing no location.
                let result = if position.timestamp().is_finite() {
                    Ok(WebPermissionOutcome::GeolocationGranted)
                } else {
                    Err(malformed(
                        "Browser geolocation callback has a nonfinite timestamp",
                    ))
                };
                settle(pending, result);
            });
        // Condition (1): native WebIDL error callbacks use the same interop ABI.
        let rejected = Closure::<dyn FnMut(JsValue, PositionError)>::new(
            |key: JsValue, error: PositionError| {
                let Some(pending) = take(&key) else {
                    return;
                };
                let kind = pending.registration.token.kind;
                let error = WebPermissionError {
                    code: Some(u32::from(error.code())),
                    name: None,
                    message: error.message(),
                };
                settle(pending, Ok(WebPermissionOutcome::Rejected { kind, error }));
            },
        );
        Self { position, rejected }
    }

    fn bind(&self, key: CallbackKey) -> (Function, Function) {
        let key = JsValue::from_str(&key.encode());
        let position: &Function = self.position.as_ref().unchecked_ref();
        let rejected: &Function = self.rejected.as_ref().unchecked_ref();
        // Native bind returns genuine functions. The WebIDL watch method erases
        // their Rust arity marker while the bound key occupies the first argument.
        (
            position.bind1(&JsValue::UNDEFINED, &key).unchecked_into(),
            rejected.bind1(&JsValue::UNDEFINED, &key).unchecked_into(),
        )
    }
}

pub(super) fn register(
    registration: &Rc<Registration>,
    completion: &Rc<Completion>,
) -> std::io::Result<(CallbackKey, Function, Function)> {
    let key = SEQUENCE.with(|sequence| sequence.borrow_mut().next())?;
    let (position, rejected) = DISPATCHERS.with(|dispatchers| {
        dispatchers
            .borrow_mut()
            .get_or_insert_with(Dispatchers::new)
            .bind(key)
    });
    PENDING.with(|pending| {
        let old = pending.borrow_mut().insert(
            key,
            Pending {
                registration: Rc::clone(registration),
                completion: Rc::downgrade(completion),
            },
        );
        assert!(old.is_none(), "realm geolocation keys never repeat");
    });
    Ok((key, position, rejected))
}

pub(super) fn registration_failed(key: CallbackKey) {
    // A synchronous registration exception has no native watch/callback task.
    PENDING.with(|pending| pending.borrow_mut().remove(&key));
}

pub(super) fn dispatcher_count() -> usize {
    DISPATCHERS.with(|dispatchers| if dispatchers.borrow().is_some() { 2 } else { 0 })
}
