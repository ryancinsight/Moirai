//! Real SendInput reaches installed controller callbacks with Handled unchanged.

use crate::windows::window::focus::NativeFocusTarget;
use crate::windows::{
    webview::{
        WebViewHost,
        tests::{
            TestPackage,
            startup::{Trace, tracing},
        },
    },
    window::{NativeWindow, WindowConfig, WindowVisibility},
};
use std::time::{Duration, Instant};
use webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC;
use windows::Win32::UI::{
    Input::KeyboardAndMouse::{
        GetFocus, GetKeyboardLayout, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
        KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MAPVK_VK_TO_VSC_EX,
        MapVirtualKeyExW, SendInput, SetFocus,
    },
    WindowsAndMessaging::{GetWindowThreadProcessId, IsChild, SetForegroundWindow},
};

#[test]
#[ignore = "requires an installed WebView2 runtime, visible desktop and explicit native observations"]
fn installed_runtime_observes_controller_input_without_handling() {
    observe_controller_input(false);
}

#[test]
#[ignore = "requires an installed WebView2 runtime, visible desktop and explicit native observations"]
fn installed_runtime_restores_controller_input_after_native_parent_focus() {
    observe_controller_input(true);
}

mod late;

fn observe_controller_input(native_parent_focus: bool) -> Trace {
    let (trace, _subscriber) = tracing();
    let _test = tracing::info_span!("controller_native_input").entered();
    let (mut package, mut host) = tracing::info_span!("arrange").in_scope(|| {
        let mut package = TestPackage::create_with_script(
            br#"<!doctype html><meta charset="utf-8"><input autofocus value="actual seed"><p id="host-status">native input</p><output id="result">unsubmitted</output>"#,
        );
        package.permit_cleanup(false);
        let window = NativeWindow::new(
            &WindowConfig::with_visibility("Moirai actual controller input", 480, 320, WindowVisibility::Visible)
                .expect("window configuration"),
        ).expect("native window");
        let host = WebViewHost::new(window, package.fresh_config()).expect("installed WebView2");
        assert!(host.startup.is_some(), "fixture selection must install the real diagnostic");
        (package, host)
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    tracing::info_span!("act").in_scope(|| {
        // These are this fixture's explicit input actions. The observer itself
        // never sets focus or Handled. No page script supplies input or results.
        unsafe {
            assert!(
                SetForegroundWindow(host.window.hwnd).as_bool(),
                "actual foreground activation"
            );
            host.controller
                .as_ref()
                .expect("live controller")
                .MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC)
                .expect("real controller focus");
        }
        while !trace
            .observations()
            .iter()
            .any(|r| r["event"]["name"] == "GotFocus")
        {
            assert!(
                Instant::now() < deadline,
                "actual controller focus: {:?}",
                trace.observations()
            );
            host.wait_events(Duration::from_millis(25))
                .expect("owner pump");
        }
    });
    if native_parent_focus {
        tracing::info_span!("act").in_scope(|| {
            // A native focus request to this controller's parent must enter
            // the controller synchronously. No test-side MoveFocus follows
            // this transition; subsequent input uses the actual recipient.
            unsafe {
                SetFocus(Some(host.window.hwnd)).expect("actual parent focus request");
                let recipient = GetFocus();
                assert!(
                    IsChild(host.window.hwnd, recipient).as_bool(),
                    "native parent focus must restore its controller: parent={:?}, recipient={recipient:?}",
                    host.window.hwnd
                );
            }
        });
    }
    let keys = [0x11_u32, 0x41, 0x41, 0x11, 0x1b, 0x1b];
    let releases = [false, false, true, true, false, true];
    let records = tracing::info_span!("arrange").in_scope(|| {
        // SAFETY: immediate reads identify the actual recipient's layout.
        let layout = unsafe {
            let focus = GetFocus();
            assert!(!focus.is_invalid());
            GetKeyboardLayout(GetWindowThreadProcessId(focus, None))
        };
        std::array::from_fn::<_, 6, _>(|index| {
            // SAFETY: the SDK maps one admitted virtual key using that HKL.
            let mapped = unsafe { MapVirtualKeyExW(keys[index], MAPVK_VK_TO_VSC_EX, Some(layout)) };
            assert_ne!(mapped & 0xff, 0);
            assert!(matches!(mapped >> 8, 0 | 0xe0));
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wScan: u16::try_from(mapped & 0xff).expect("scan byte"),
                        dwFlags: KEYEVENTF_SCANCODE
                            | if mapped >> 8 == 0xe0 {
                                KEYEVENTF_EXTENDEDKEY
                            } else {
                                KEYBD_EVENT_FLAGS::default()
                            }
                            | if releases[index] {
                                KEYEVENTF_KEYUP
                            } else {
                                KEYBD_EVENT_FLAGS::default()
                            },
                        ..Default::default()
                    },
                },
            }
        })
    });
    tracing::info_span!("act").in_scope(|| {
        // SAFETY: every record is a real immediate INPUT with its ABI size;
        // the single ordered sequence includes every held key's release.
        let sent = unsafe {
            SendInput(
                &records,
                i32::try_from(size_of::<INPUT>()).expect("INPUT size"),
            )
        };
        if sent != 6 {
            // Release only a down admitted without its matching up, then keep
            // the original partial delivery as the primary failed assertion.
            for (down, up) in [(0, 3), (1, 2), (4, 5)] {
                if sent > down && sent <= up {
                    unsafe {
                        SendInput(
                            &records[up as usize..=up as usize],
                            size_of::<INPUT>() as i32,
                        );
                    }
                }
            }
        }
        assert_eq!(sent, 6, "actual native insertion count");
        while trace
            .observations()
            .iter()
            .filter(|r| r["event"]["name"] == "AcceleratorKeyPressed")
            .count()
            < 6
        {
            assert!(
                Instant::now() < deadline,
                "actual controller keys: {:?}",
                trace.observations()
            );
            host.wait_events(Duration::from_millis(25))
                .expect("owner input pump");
        }
    });
    tracing::info_span!("assert").in_scope(|| {
        let observations = trace.observations();
        let actual: Vec<_> = observations
            .iter()
            .filter(|r| r["event"]["name"] == "AcceleratorKeyPressed")
            .collect();
        assert_eq!(actual.len(), keys.len());
        for (index, record) in actual.iter().enumerate() {
            assert_eq!(record["event"]["virtual_key"], keys[index]);
            assert_eq!(record["event"]["event_kind"], i32::from(releases[index]));
            assert_eq!(record["event"]["handled"], false);
            // The callback's lParam must preserve the real physical scan and
            // release bit. Signed JSON is reinterpreted as the Win32 bits.
            let bits = i32::try_from(record["event"]["lparam"].as_i64().expect("signed lParam"))
                .expect("Int32")
                .cast_unsigned();
            assert_eq!((bits >> 31) != 0, releases[index]);
            assert_eq!(
                (bits >> 16) & 0xff,
                u32::from(unsafe { records[index].Anonymous.ki.wScan })
            );
        }
        for record in &observations {
            assert_ne!(record["kind"], "controller_input_loss");
            assert_ne!(record["event"]["name"], "read_error");
            assert_ne!(record["kind"], "native_error");
            assert_ne!(record["kind"], "quota_exhausted");
        }
    });
    let closed_target =
        native_parent_focus.then(|| host.focus.as_ref().expect("live focus owner").clone());
    tracing::info_span!("act")
        .in_scope(|| host.close().expect("remove actual controller callbacks"));
    package.permit_cleanup(true);
    tracing::info_span!("assert").in_scope(|| {
        if let Some(target) = closed_target {
            let error = target
                .focus()
                .expect_err("closed native controller rejects focus");
            assert!(
                error
                    .to_string()
                    .contains("WebView2 MoveFocus(PROGRAMMATIC)")
            );
            let native = error
                .get_ref()
                .expect("retained native source")
                .downcast_ref::<windows::core::Error>()
                .expect("actual SDK error");
            assert!(native.code().is_err(), "actual HRESULT remains failed");
        }
        let observations = trace.observations();
        let closed = observations.last().expect("actual close metadata");
        assert_eq!(closed["kind"], "closed");
        assert_eq!(closed["closed"], true);
        assert_eq!(closed["cleanup_failed"], false);
        assert_eq!(closed["omitted"], "0");
        package
            .cleanup()
            .expect("owned profile cleanup after browser exit");
        let selection: serde_json::Value = serde_json::from_str(
            &std::env::var(crate::webview_startup::SELECTION_ENV)
                .expect("explicit installed fixture selection"),
        )
        .expect("actual selected configuration");
        if selection["late_input_after_key"].is_null() {
            assert_eq!(trace.late_bytes(), Vec::<u8>::new());
        }
    });
    trace
}
