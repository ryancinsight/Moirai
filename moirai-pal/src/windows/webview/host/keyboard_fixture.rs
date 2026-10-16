//! Controlled physical keyboard experiment; no production accelerator policy.

mod input;
mod policy;
#[path = "../../../../examples/webview_keyboard_fixture/protocol.rs"]
mod protocol;
mod session;

use super::view::WebViewHost;
use crate::windows::webview::event::WebViewHostEvent;
use crate::windows::{
    webview::tests::startup,
    window::{NativeWindow, WindowConfig, WindowEvent, WindowVisibility},
};
use policy::{Policy, Registration};
use protocol::{Command, Notice};
use serde_json::json;
use std::{
    io,
    time::{Duration, Instant},
};
use webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC;
use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;

#[test]
#[ignore = "requires original Rust payload, installed WebView2 and physical keyboard input"]
fn installed_runtime_compares_physical_shortcut_browser_policy() {
    let (trace, _subscriber) = startup::tracing();
    let policy = match std::env::var("MOIRAI_WEBVIEW_KEYBOARD_POLICY")
        .expect("explicit experiment policy")
        .as_str()
    {
        "default" => Policy::Default,
        "enable_s" => Policy::EnableS,
        "skip_s" => Policy::SkipS,
        _ => panic!("unknown keyboard experiment policy"),
    };
    let expected_observer = match std::env::var("MOIRAI_WEBVIEW_KEYBOARD_OBSERVER")
        .expect("explicit experiment observer")
        .as_str()
    {
        "selected" => true,
        "none" => false,
        _ => panic!("unknown keyboard experiment observer"),
    };
    let _test =
        tracing::info_span!("actual_keyboard_counterfactual", ?policy, expected_observer).entered();
    let deadline = Instant::now() + Duration::from_secs(30);
    let (mut package, payload) = tracing::info_span!("arrange").in_scope(session::package);
    let window = NativeWindow::new(
        &WindowConfig::with_visibility(
            "Moirai physical keyboard comparison",
            640,
            480,
            WindowVisibility::Visible,
        )
        .expect("real window configuration"),
    )
    .expect("actual native window");
    let mut host = WebViewHost::new(window, package.fresh_config()).expect("installed browser");
    let observer_selected = host.startup.is_some();
    let mut registration =
        Registration::new(host.controller.as_ref().expect("actual controller"), policy);
    let runtime = runtime_version(&host);
    let mut notices = session::Document::default();
    let mut chords = Vec::new();
    let result = tracing::info_span!("act").in_scope(|| {
        if observer_selected != expected_observer {
            return Err(io::Error::other(
                "actual startup observer selection differs",
            ));
        }
        exercise(
            &mut host,
            &registration,
            &mut notices,
            &mut chords,
            deadline,
        )
    });
    let callback_cleanup = registration.close();
    let records = registration.records.borrow().clone();
    drop(registration);
    let host_cleanup = host.close();
    let destroyed = host.poll_events().map(|events| {
        events
            .iter()
            .filter(|event| matches!(event, WebViewHostEvent::Window(WindowEvent::Destroyed)))
            .count()
    });
    drop(host);
    let released = callback_cleanup.is_ok()
        && host_cleanup.is_ok()
        && destroyed.as_ref().is_ok_and(|count| *count == 1);
    package.permit_cleanup(released);
    let profile_cleanup = package.cleanup();
    let encoded = json!({"schema":2,"policy":policy,"observer_selected":observer_selected,"payload_directory":payload.directory,"payload_files":payload.files,"runtime":runtime.as_ref().ok(),"runtime_error":runtime.as_ref().err().map(ToString::to_string),"chords":chords,"controller":records,"document":notices,
        "operation_error":result.as_ref().err().map(ToString::to_string),"callback_cleanup":callback_cleanup.as_ref().err().map(ToString::to_string),
        "host_cleanup":host_cleanup.as_ref().err().map(ToString::to_string),"destroyed":destroyed.as_ref().ok(),
        "profile_cleanup":profile_cleanup.as_ref().err().map(ToString::to_string),
        "coverage":"observed prefix through real P and closed document listeners; no keyboard/IPC drain barrier"});
    // Retain primary operation and cleanup failures before fallible publication.
    // Publication is secondary: it cannot replace an already-owned failure.
    tracing::info!(event="keyboard_counterfactual", observation=%encoded,
        operation_error=?result.as_ref().err(), callback_cleanup=?callback_cleanup.as_ref().err(),
        host_cleanup=?host_cleanup.as_ref().err(), destroyed=?destroyed.as_ref().err(),
        profile_cleanup=?profile_cleanup.as_ref().err(), "actual native keyboard result");
    let publication = publish_evidence(&encoded);
    if let Err(error) = &publication {
        tracing::error!(event="keyboard_evidence_publication_failed", %error,
            "secondary evidence publication failure");
    }
    tracing::info_span!("assert").in_scope(|| {
        result.expect("actual physical keyboard operation");
        assert!(
            !runtime
                .expect("actual controller environment version")
                .is_empty()
        );
        callback_cleanup.expect("remove additional fixture callback");
        host_cleanup.expect("actual browser close");
        assert_eq!(destroyed.expect("actual destruction drain"), 1);
        profile_cleanup.expect("actual profile/package cleanup");
        assert!(!package.directory.exists() && !package.profile.exists());
        assert_values(policy, &records, &notices);
        assert_strokes(&records, &chords);
        for record in trace.observations() {
            assert!(
                !matches!(
                    record["kind"].as_str(),
                    Some("native_error" | "state_error" | "process_failed" | "quota_exhausted")
                ),
                "{record}"
            );
        }
        publication.expect("retain actual experiment values");
    });
}

fn publish_evidence(encoded: &serde_json::Value) -> io::Result<()> {
    let output = std::env::var_os("MOIRAI_WEBVIEW_KEYBOARD_EVIDENCE").ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "missing exclusive evidence path")
    })?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer(&mut file, encoded).map_err(io::Error::other)
}

fn exercise(
    host: &mut WebViewHost,
    registration: &Registration,
    notices: &mut session::Document,
    chords: &mut Vec<input::Chord>,
    deadline: Instant,
) -> io::Result<()> {
    session::until(host, registration, notices, deadline, |rows| {
        matches!(rows.first(), Some(Notice::Ready {}))
    })?;
    // SAFETY: explicit test actions activate the owned native window/controller.
    unsafe {
        if !SetForegroundWindow(host.window.hwnd).as_bool() {
            return Err(io::Error::other(
                "SetForegroundWindow returned false for the owned fixture window",
            ));
        }
        host.controller
            .as_ref()
            .expect("live controller")
            .MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC)
            .map_err(io::Error::other)?;
    }
    input::insert(host.window.hwnd, 68, chords)?;
    session::until(host, registration, notices, deadline, |rows| {
        has_key(rows, "KeyD")
    })?;
    await_released(host, registration, notices, deadline, 68)?;
    input::insert(host.window.hwnd, 83, chords)?;
    await_released(host, registration, notices, deadline, 83)?;
    input::insert(host.window.hwnd, 80, chords)?;
    session::until(host, registration, notices, deadline, |rows| {
        has_key(rows, "KeyP")
    })?;
    await_released(host, registration, notices, deadline, 80)?;
    host.post_json(&serde_json::to_string(&Command::Close {}).map_err(io::Error::other)?)?;
    session::until(host, registration, notices, deadline, |rows| {
        matches!(rows.last(), Some(Notice::Closed { .. }))
    })
}

fn await_released(
    host: &mut WebViewHost,
    registration: &Registration,
    notices: &mut session::Document,
    deadline: Instant,
    key: u32,
) -> io::Result<()> {
    loop {
        let released = {
            let records = registration.records.borrow();
            records
                .iter()
                .any(|row| row.key == Some(key) && matches!(row.kind, Some(1 | 3)))
                && records.last().is_some_and(|row| {
                    row.held
                        .is_some_and(|held| held.into_iter().all(|value| value >= 0))
                })
        };
        if released {
            return Ok(());
        }
        session::pump(host, registration, notices, deadline)?;
    }
}

fn runtime_version(host: &WebViewHost) -> io::Result<String> {
    let mut version = windows::core::PWSTR::null();
    // SAFETY: actual environment transfers one CoTaskMem string; the canonical
    // bounded decoder frees it after reading, including decoding failures.
    unsafe {
        host.environment
            .as_ref()
            .expect("actual environment")
            .BrowserVersionString(&mut version)
            .map_err(io::Error::other)?;
        super::text::read_task_mem_message(version).map_err(io::Error::other)
    }
}

fn has_key(rows: &[Notice], code: &str) -> bool {
    rows.iter().any(|row| matches!(row, Notice::Key {keyboard, ..} if keyboard.code == code && !keyboard.ctrl && keyboard.alt && keyboard.shift && !keyboard.meta && !keyboard.repeat && keyboard.trusted))
}

fn assert_values(policy: Policy, records: &[policy::Attempt], notices: &[Notice]) {
    assert!(notices.len() >= 2);
    assert!(matches!(notices.first(), Some(Notice::Ready {})));
    assert!(matches!(notices.last(), Some(Notice::Closed { .. })));
    assert!(
        notices[1..notices.len() - 1]
            .iter()
            .all(|row| matches!(row, Notice::Key { .. }))
    );
    let actual: Vec<_> = notices
        .iter()
        .filter_map(|row| {
            if let Notice::Key { sequence, keyboard } = row {
                Some((*sequence, keyboard))
            } else {
                None
            }
        })
        .collect();
    assert!(has_key(notices, "KeyD") && has_key(notices, "KeyP"));
    for (index, (sequence, _)) in actual.iter().enumerate() {
        assert_eq!(
            usize::try_from(*sequence).expect("bounded sequence"),
            index + 1
        );
    }
    for record in records {
        assert_eq!(record.handled_before, record.handled_after);
    }
    let selected: Vec<_> = records
        .iter()
        .filter(|row| policy::matches_system(row))
        .collect();
    assert_eq!(
        selected.len(),
        1,
        "the actual fresh S-down intervention domain must execute once"
    );
    assert_eq!(selected[0].browser_before, Some(true));
    let interventions: Vec<_> = records
        .iter()
        .filter(|row| row.setter_succeeded.is_some())
        .collect();
    match policy {
        Policy::Default => {
            assert!(interventions.is_empty());
            assert_eq!(selected[0].browser_after, Some(true));
        }
        Policy::EnableS | Policy::SkipS => {
            assert_eq!(interventions.len(), 1);
            assert!(policy::matches_system(interventions[0]));
            assert_eq!(interventions[0].browser_before, Some(true));
            assert_eq!(
                interventions[0].browser_after,
                Some(policy == Policy::EnableS)
            );
            assert_eq!(interventions[0].setter_succeeded, Some(true));
            if policy == Policy::SkipS {
                assert!(
                    has_key(notices, "KeyS"),
                    "actual S must reach the disabled-browser document"
                );
            }
        }
    }
    match notices.last().expect("actual terminal notice") {
        Notice::Closed {
            key_events,
            publication_error,
            bridge_retained,
            dom_active,
            dom_retained,
        } => {
            assert_eq!(
                usize::try_from(*key_events).expect("bounded count"),
                actual.len()
            );
            assert_eq!(publication_error, &None);
            assert_eq!((*bridge_retained, *dom_active, *dom_retained), (0, 0, 0));
        }
        other => panic!("actual closed notice missing: {other:?}"),
    }
}

fn assert_strokes(records: &[policy::Attempt], chords: &[input::Chord]) {
    assert_eq!(chords.len(), 3);
    for (chord, letter) in chords.iter().zip([68, 83, 80]) {
        assert_eq!(chord.letter, letter);
        assert_eq!(chord.admitted, Some(6));
        assert_eq!(chord.insertion_error, None);
        assert!(chord.releases.is_empty());
        let letter_records: Vec<_> = records
            .iter()
            .filter(|row| row.key == Some(letter))
            .collect();
        assert_eq!(letter_records.len(), 2, "exact original letter down/up");
        for (record, release) in letter_records.into_iter().zip([false, true]) {
            assert!(matches!(
                (record.kind, release),
                (Some(0 | 2), false) | (Some(1 | 3), true)
            ));
            let bits = record
                .lparam
                .expect("actual supplied lParam")
                .cast_unsigned();
            assert_eq!((bits >> 31) != 0, release);
            assert_eq!((bits >> 16) & 0xff, u32::from(chord.strokes[2].scan));
            assert_eq!((bits >> 24) & 1 != 0, chord.strokes[2].flags & 1 != 0);
            assert_eq!(record.handled_before, Some(false));
            assert_eq!(record.handled_after, Some(false));
            let held = record.held.expect("actual owner key states");
            assert!(held[0] >= 0 && held[1] < 0 && held[2] < 0 && held[3] >= 0 && held[4] >= 0);
        }
    }
}
