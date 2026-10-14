//! Actual native-object callbacks with deferred Rust-owned close/remount.

use std::{cell::Cell, io, rc::Rc};

use futures::{StreamExt, channel::mpsc};
use moirai_pal::wasm::{WebViewBridge, WebViewMessageListener};
use wasm_bindgen::prelude::*;

use crate::protocol::{Command, Notice};

type Incoming = mpsc::Receiver<io::Result<String>>;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let bridge =
        Rc::new(WebViewBridge::current().map_err(|error| JsValue::from_str(&error.to_string()))?);
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = run(bridge).await {
            wasm_bindgen::throw_str(&format!("Rust native bridge fixture: {error}"));
        }
    });
    Ok(())
}

fn attach(
    bridge: &WebViewBridge,
    calls: &Rc<Cell<u32>>,
) -> io::Result<(WebViewMessageListener, Incoming)> {
    // One sender reserves exactly one slot; it is never cloned. Only the
    // native request/response protocol can admit the next outstanding value.
    let (mut sender, incoming) = mpsc::channel(0);
    let calls = Rc::clone(calls);
    let listener = bridge.listen(move |value| {
        calls.set(
            calls
                .get()
                .checked_add(1)
                .expect("finite fixture callbacks"),
        );
        sender
            .try_send(value)
            .expect("one outstanding actual native command");
    })?;
    Ok((listener, incoming))
}

async fn next(incoming: &mut Incoming) -> io::Result<(String, Command)> {
    let json = incoming
        .next()
        .await
        .ok_or_else(|| io::Error::other("actual native listener closed before its command"))??;
    let command = serde_json::from_str(&json)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok((json, command))
}

fn send(bridge: &WebViewBridge, value: &Notice) -> io::Result<()> {
    let json = serde_json::to_string(value).map_err(io::Error::other)?;
    bridge.post_json(&json)
}

fn unexpected(command: &Command) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("unexpected actual native command: {command:?}"),
    )
}

async fn echo(
    bridge: &WebViewBridge,
    incoming: &mut Incoming,
    mount: u8,
    calls: &Cell<u32>,
    old_calls: &Cell<u32>,
) -> io::Result<()> {
    let (_, command) = next(incoming).await?;
    let Command::Echo {
        mount: actual,
        identity,
        patient,
        values,
    } = command
    else {
        return Err(unexpected(&command));
    };
    if actual != mount {
        return Err(io::Error::other("actual echo mount differs"));
    }
    let total = values.into_iter().try_fold(0_u32, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| io::Error::other("actual fixture sum overflows"))
    })?;
    send(
        bridge,
        &Notice::Echo {
            mount,
            identity,
            patient,
            total,
            calls: calls.get(),
            old_calls: old_calls.get(),
        },
    )
}

async fn boundary(bridge: &WebViewBridge, incoming: &mut Incoming) -> io::Result<()> {
    let (json, command) = next(incoming).await?;
    if !matches!(command, Command::Boundary { mount: 1, .. }) {
        return Err(unexpected(&command));
    }
    if json.len() != WebViewBridge::message_byte_limit() {
        return Err(io::Error::other(
            "actual boundary input does not equal the byte limit",
        ));
    }
    let excessive = format!("{json} ");
    let error = bridge
        .post_json(&excessive)
        .expect_err("one byte beyond the actual bridge limit");
    if error.kind() != io::ErrorKind::InvalidInput {
        return Err(error);
    }
    send(
        bridge,
        &Notice::Rejected {
            mount: 1,
            bytes: excessive.len(),
            error_kind: format!("{:?}", error.kind()),
            error: error.to_string(),
        },
    )?;
    // Identical actual received JSON exercises the positive native send bound.
    bridge.post_json(&json)
}

async fn close(
    bridge: &WebViewBridge,
    listener: &mut WebViewMessageListener,
    incoming: &mut Incoming,
    mount: u8,
    calls: &Cell<u32>,
    old_calls: &Cell<u32>,
) -> io::Result<()> {
    let (_, command) = next(incoming).await?;
    if !matches!(command, Command::Close { mount: actual } if actual == mount) {
        return Err(unexpected(&command));
    }
    listener.close()?;
    send(
        bridge,
        &Notice::Closed {
            mount,
            calls: calls.get(),
            old_calls: old_calls.get(),
            retained: WebViewBridge::retained_cleanup_count(),
        },
    )
}

async fn run(bridge: Rc<WebViewBridge>) -> io::Result<()> {
    let retired = Rc::new(Cell::new(0));
    let absent = Cell::new(0);
    let (mut first, mut incoming) = attach(&bridge, &retired)?;
    send(
        &bridge,
        &Notice::Ready {
            mount: 1,
            old_calls: 0,
        },
    )?;
    echo(&bridge, &mut incoming, 1, &retired, &absent).await?;
    boundary(&bridge, &mut incoming).await?;
    close(&bridge, &mut first, &mut incoming, 1, &retired, &absent).await?;
    drop((first, incoming));

    let calls = Rc::new(Cell::new(0));
    let (mut second, mut incoming) = attach(&bridge, &calls)?;
    send(
        &bridge,
        &Notice::Ready {
            mount: 2,
            old_calls: retired.get(),
        },
    )?;
    let (_, command) = next(&mut incoming).await?;
    let Command::Probe { mount: 2, identity } = command else {
        return Err(unexpected(&command));
    };
    send(
        &bridge,
        &Notice::Probe {
            mount: 2,
            identity,
            calls: calls.get(),
            old_calls: retired.get(),
        },
    )?;
    echo(&bridge, &mut incoming, 2, &calls, &retired).await?;
    close(&bridge, &mut second, &mut incoming, 2, &calls, &retired).await
}
