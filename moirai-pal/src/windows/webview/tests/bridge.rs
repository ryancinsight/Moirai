//! Real Rust/WASM native-object admission, byte bounds and listener retirement.

mod session;

// One fixture wire definition is compiled into both actual Rust endpoints.
#[path = "../../../../examples/webview_bridge_fixture/protocol.rs"]
mod protocol;

use super::{
    NativeWindow, TestPackage, WebViewEvent, WebViewHost, WebViewHostEvent, WindowConfig,
    WindowVisibility, startup,
};
use crate::{webview_limits::MAX_WEBVIEW_MESSAGE_BYTES, windows::window::WindowEvent};
use protocol::{Command, Notice};
use session::Session;
use std::{io, time::Instant};

#[test]
#[ignore = "requires the actual compiled Rust payload and installed WebView2 runtime"]
fn installed_runtime_roundtrips_rust_bridge_and_retires_closed_listeners() {
    let (trace, _subscriber) = startup::tracing();
    let _test = tracing::info_span!("actual_rust_native_bridge").entered();
    let (package, mut session) = tracing::info_span!("arrange").in_scope(|| {
        let package = session::package();
        let config = package.fresh_config();
        let window = NativeWindow::new(
            &WindowConfig::with_visibility(
                "Moirai compiled Rust bridge",
                480,
                320,
                WindowVisibility::Visible,
            )
            .expect("actual fixture window configuration"),
        )
        .expect("actual fixture window");
        let deadline = Instant::now() + config.wait();
        let host = WebViewHost::new(window, config).expect("installed actual WebView2 runtime");
        (package, Session::new(host, trace, deadline))
    });
    tracing::info_span!("act").in_scope(|| {
        session.ready(1, 0);
        roundtrip(&mut session, 1, 1, 1, 0);
        boundary(&mut session);
        session.close_listener(1, 3, 0);
        session.ready(2, 3);
        session.send(&Command::Probe {
            mount: 2,
            identity: "9007199254740993".to_owned(),
        });
        match session.next() {
            Notice::Probe {
                mount,
                identity,
                calls,
                old_calls,
            } => {
                assert_eq!((mount, calls, old_calls), (2, 1, 3));
                assert_eq!(identity, "9007199254740993");
            }
            actual => panic!("actual new-listener probe: {actual:?}"),
        }
        roundtrip(&mut session, 2, 3, 2, 3);
        session.close_listener(2, 3, 3);
    });
    tracing::info_span!("assert").in_scope(|| {
        session.finish();
        package
            .cleanup()
            .expect("actual compiled-payload/profile release");
        assert!(!package.directory.exists() && !package.profile.exists());
    });
}

fn roundtrip(session: &mut Session, mount: u8, factor: u32, calls: u32, old: u32) {
    let identity = if mount == 1 {
        "18446744073709551615"
    } else {
        "9007199254740993"
    };
    let patient = format!("患者 / Zoë / λ🦀 / mount {mount}");
    session.send(&Command::Echo {
        mount,
        identity: identity.to_owned(),
        patient: patient.clone(),
        values: [factor, factor << 8, factor << 16],
    });
    match session.next() {
        Notice::Echo {
            mount: actual,
            identity: actual_id,
            patient: actual_patient,
            total,
            calls: actual_calls,
            old_calls,
        } => {
            assert_eq!(actual, mount);
            assert_eq!(actual_id, identity);
            assert_eq!(actual_patient, patient);
            // Independent closed form for the actual Rust input reduction.
            assert_eq!(total, factor * (1 + 256 + 65_536));
            assert_eq!((actual_calls, old_calls), (calls, old));
            tracing::info!(
                event = "rust_bridge_roundtrip",
                mount,
                identity = actual_id,
                patient = actual_patient,
                total,
                actual_calls,
                old_calls,
                "Actual installed-runtime Rust bridge values"
            );
        }
        actual => panic!("actual computed Rust bridge reply: {actual:?}"),
    }
}

fn boundary(session: &mut Session) {
    let prefix = "λ🦀";
    let empty = Command::Boundary {
        mount: 1,
        text: String::new(),
    };
    let overhead = serde_json::to_vec(&empty)
        .expect("actual boundary overhead")
        .len();
    let text = format!(
        "{prefix}{}",
        "a".repeat(MAX_WEBVIEW_MESSAGE_BYTES - overhead - prefix.len())
    );
    let json = serde_json::to_string(&Command::Boundary {
        mount: 1,
        text: text.clone(),
    })
    .expect("actual inclusive JSON");
    assert_eq!(json.len(), MAX_WEBVIEW_MESSAGE_BYTES);
    let excessive = format!("{json} ");
    assert_eq!(excessive.len(), MAX_WEBVIEW_MESSAGE_BYTES + 1);
    let error = session
        .host
        .post_json(&excessive)
        .expect_err("native one-byte excess");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    session
        .host
        .post_json(&json)
        .expect("actual inclusive native post");
    match session.next() {
        Notice::Rejected {
            mount,
            bytes,
            error_kind,
            error,
        } => {
            assert_eq!(mount, 1);
            assert_eq!(bytes, MAX_WEBVIEW_MESSAGE_BYTES + 1);
            assert_eq!(error_kind, "InvalidInput");
            assert!(error.contains("bounded byte limit"), "{error}");
        }
        actual => panic!("actual WASM one-byte rejection: {actual:?}"),
    }
    match session.next() {
        Notice::Boundary {
            mount,
            text: actual,
        } => {
            assert_eq!(mount, 1);
            assert_eq!(actual, text);
        }
        actual => panic!("actual inclusive WASM post: {actual:?}"),
    }
}
