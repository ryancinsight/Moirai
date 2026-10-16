//! Observe actual keyboard callbacks; native Close retires both listener owners.

use std::{
    cell::{Cell, RefCell},
    fmt, io,
    rc::Rc,
};

use futures::channel::oneshot;
use moirai_pal::wasm::{WebDocument, WebEvent, WebEventListener, WebViewBridge};
use wasm_bindgen::prelude::*;

use crate::protocol::{Command, MAX_KEY_RECORDS, Notice, SUCCESSFUL_CLOSED_BYTES};

type Incoming = oneshot::Receiver<io::Result<String>>;

#[derive(Default)]
struct State {
    key_events: Cell<u32>,
    encoded_bytes: Cell<usize>,
    closing: Cell<bool>,
    publication_error: RefCell<Option<io::Error>>,
}

impl State {
    fn fail(&self, error: io::Error) {
        self.publication_error.borrow_mut().get_or_insert(error);
    }

    fn publish(&self, bridge: &WebViewBridge, notice: &Notice) -> io::Result<()> {
        let encoded = serde_json::to_string(notice).map_err(io::Error::other)?;
        let next = self
            .encoded_bytes
            .get()
            .checked_add(encoded.len())
            .ok_or_else(|| {
                io::Error::other("actual keyboard fixture encoded byte count overflows")
            })?;
        let bound = WebViewBridge::message_byte_limit();
        let capacity = if matches!(notice, Notice::Closed { .. }) {
            bound
        } else {
            bound.checked_sub(SUCCESSFUL_CLOSED_BYTES).ok_or_else(|| {
                io::Error::other("successful keyboard close reserve exceeds bridge capacity")
            })?
        };
        if next > capacity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "actual keyboard fixture notices exceed aggregate bridge byte capacity",
            ));
        }
        // Reserve before the actual call. Reentrant callbacks cannot reuse
        // these bytes, and a failed attempted publication never refunds them.
        self.encoded_bytes.set(next);
        bridge.post_json(&encoded)
    }

    fn observe(&self, bridge: &WebViewBridge, event: &WebEvent) {
        if self.closing.get() {
            return;
        }
        let Some(sequence) = self.key_events.get().checked_add(1) else {
            self.fail(io::Error::other("actual keyboard callback count overflows"));
            return;
        };
        self.key_events.set(sequence);
        if self.publication_error.borrow().is_some() {
            return;
        }
        let result = (|| {
            if sequence > MAX_KEY_RECORDS {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "actual keyboard callback exceeds the finite key-notice capacity",
                ));
            }
            let keyboard = event.keyboard_metadata()?.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "keydown has no keyboard metadata",
                )
            })?;
            self.publish(
                bridge,
                &Notice::Key {
                    sequence,
                    keyboard: keyboard.into(),
                },
            )
        })();
        if let Err(error) = result {
            self.fail(error);
        }
    }
}

/// Each actual failure remains owned until after both teardown attempts.
#[derive(Debug, Default)]
struct Failures {
    publication: Option<io::Error>,
    operation: Option<io::Error>,
    keyboard_cleanup: Option<io::Error>,
    bridge_cleanup: Option<io::Error>,
    closed_publication: Option<io::Error>,
}

impl Failures {
    fn entries(&self) -> [(&str, Option<&io::Error>); 5] {
        [
            ("keyboard publication", self.publication.as_ref()),
            ("fixture operation", self.operation.as_ref()),
            ("keyboard cleanup", self.keyboard_cleanup.as_ref()),
            ("bridge cleanup", self.bridge_cleanup.as_ref()),
            ("closed publication", self.closed_publication.as_ref()),
        ]
    }

    fn result(self) -> io::Result<()> {
        let first_kind = self
            .entries()
            .into_iter()
            .find_map(|(_, error)| error.map(io::Error::kind));
        if let Some(kind) = first_kind {
            Err(io::Error::new(kind, self))
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for Failures {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        for (operation, error) in self.entries() {
            if let Some(error) = error {
                write!(formatter, "{separator}{operation}: {error}")?;
                separator = "; ";
            }
        }
        Ok(())
    }
}

impl std::error::Error for Failures {
    // Type erasure belongs to the standard heterogeneous error-source ABI
    // (condition 2), outside keyboard observation or publication work.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.entries()
            .into_iter()
            .find_map(|(_, error)| error.map(|error| error as &(dyn std::error::Error + 'static)))
    }
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let bridge = Rc::new(WebViewBridge::current().map_err(js_error)?);
    let document = WebDocument::current().map_err(js_error)?;
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = run(bridge, document).await {
            wasm_bindgen::throw_str(&format!("Rust keyboard fixture: {error}"));
        }
    });
    Ok(())
}

fn js_error(error: io::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn attach_native(
    bridge: &WebViewBridge,
    state: &Rc<State>,
) -> io::Result<(moirai_pal::wasm::WebViewMessageListener, Incoming)> {
    // There is exactly one native command: Close. The one-shot owns its
    // actual reply until the deferred task runs; no callback closes itself.
    let (sender, incoming) = oneshot::channel();
    let mut sender = Some(sender);
    let state = Rc::clone(state);
    let listener = bridge.listen(move |value| {
        if state.closing.get() {
            return;
        }
        match sender.take() {
            Some(sender) => {
                if let Err(value) = sender.send(value) {
                    state.fail(value.err().unwrap_or_else(|| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "native Close receiver is absent")
                    }));
                }
            }
            None => state.fail(io::Error::new(
                io::ErrorKind::InvalidData,
                "native keyboard fixture received another command after its one Close slot",
            )),
        }
    })?;
    Ok((listener, incoming))
}

async fn run(bridge: Rc<WebViewBridge>, document: WebDocument) -> io::Result<()> {
    let state = Rc::new(State::default());
    let (mut native, incoming) = attach_native(&bridge, &state)?;
    let mut keyboard = None;
    let operation: io::Result<()> = async {
        let captured_state = Rc::clone(&state);
        let captured_bridge = Rc::clone(&bridge);
        keyboard = Some(document.add_event_listener("keydown", move |event| {
            captured_state.observe(&captured_bridge, &event);
        })?);
        state.publish(&bridge, &Notice::Ready {})?;
        let encoded = incoming
            .await
            .map_err(|error| io::Error::new(io::ErrorKind::BrokenPipe, error))??;
        let Command::Close {} = serde_json::from_str::<Command>(&encoded)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(())
    }
    .await;
    // The oneshot wakes this task after the actual native callback returns.
    // Stop new sampling before either fallible removal; attempt both domains.
    state.closing.set(true);
    let keyboard_cleanup = keyboard
        .as_mut()
        .and_then(|listener| listener.close().err());
    let bridge_cleanup = native.close().err();
    let publication = state.publication_error.borrow_mut().take();
    let mut failures = Failures {
        publication,
        operation: operation.err(),
        keyboard_cleanup,
        bridge_cleanup,
        closed_publication: None,
    };
    if failures.operation.is_none()
        && failures.keyboard_cleanup.is_none()
        && failures.bridge_cleanup.is_none()
    {
        let notice = Notice::Closed {
            key_events: state.key_events.get(),
            publication_error: failures.publication.as_ref().map(ToString::to_string),
            bridge_retained: WebViewBridge::retained_cleanup_count(),
            dom_active: WebEventListener::active_count(),
            dom_retained: WebEventListener::retained_cleanup_count(),
        };
        failures.closed_publication = state.publish(&bridge, &notice).err();
    }
    failures.result()
}
