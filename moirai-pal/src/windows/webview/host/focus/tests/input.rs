//! Separate physical select-all and exact UTF-16 packet batches.
//!
//! One initial scope and sixteen replacement scopes each admit at most the
//! existing 256 records / 64 KiB encoded payload. Sticky exhaustion is explicit;
//! a fixed terminal summary retains actual operation counts after exhaustion.
//! Total encoded payload is at most 17 * 64 KiB, with at most 17 quota and 17
//! fixed-size terminal records (and one encoding-error record per scope).
//! These INFO receipts never enter the startup DEBUG observation parser. A
//! forcibly terminated child cannot emit a Drop summary; its last admitted
//! operation-start remains an observation, not proof of the blocked call.

use super::ObservationMode;
use crate::webview_startup::{RecordAdmission, TraceBudget};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Instant};
use windows::Win32::UI::{
    Input::KeyboardAndMouse::{
        GetKeyboardLayout, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
        KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, KEYEVENTF_UNICODE,
        MAPVK_VK_TO_VSC_EX, MapVirtualKeyExW, SendInput,
    },
    WindowsAndMessaging::{GUITHREADINFO, GetGUIThreadInfo, GetWindowThreadProcessId},
};

const TARGET: &str = "moirai_pal::native_uia_fixture";

/// Bounded fixture evidence; it has no authority over native call results.
pub(super) struct Observations {
    mode: ObservationMode,
    replacement: usize,
    field: &'static str,
    started: Instant,
    budget: TraceBudget,
    counts: BTreeMap<&'static str, usize>,
    encoding_failed: bool,
}

impl Observations {
    pub(super) fn new(mode: ObservationMode, replacement: usize) -> Self {
        Self {
            mode,
            replacement,
            field: "",
            started: Instant::now(),
            budget: TraceBudget::default(),
            counts: BTreeMap::new(),
            encoding_failed: false,
        }
    }

    pub(super) fn field(&mut self, field: &'static str) {
        self.field = field;
    }

    pub(super) fn record(&mut self, value: Value) {
        if self.encoding_failed {
            return;
        }
        let record = json!({"mode":self.mode.as_str(), "replacement":self.replacement,
            "field":self.field, "elapsed_ns":self.started.elapsed().as_nanos().to_string(),
            "details":value});
        match self.budget.record(&record) {
            Ok(RecordAdmission::Admitted(encoded)) => {
                tracing::info!(target: TARGET, event = "uia_fixture_observation", observation = %encoded);
            }
            Ok(RecordAdmission::Exhausted { first: true }) => {
                tracing::info!(target: TARGET, event = "uia_fixture_quota_exhausted", mode = self.mode.as_str(), replacement = self.replacement);
            }
            Ok(RecordAdmission::Exhausted { first: false }) => {}
            Err(error) => {
                self.encoding_failed = true;
                tracing::error!(target: TARGET, %error, "UIA evidence encoding failed");
            }
        }
    }

    pub(super) fn operation<T>(&mut self, name: &'static str, call: impl FnOnce() -> T) -> T {
        let count = self.counts.entry(name).or_default();
        *count = count
            .checked_add(1)
            .expect("bounded native operation count");
        let query = *count;
        self.record(json!({"operation":name, "stage":"start", "query":query}));
        let result = call();
        self.record(json!({"operation":name, "stage":"returned", "query":query}));
        result
    }
}

impl Drop for Observations {
    fn drop(&mut self) {
        let quota = self.budget.snapshot();
        tracing::info!(target: TARGET, event = "uia_fixture_scope_summary", mode = self.mode.as_str(), replacement = self.replacement,
            field = self.field, operations = ?self.counts, records = quota.records, bytes = quota.bytes,
            omitted = quota.omitted, exhausted = quota.exhausted, encoding_failed = self.encoding_failed,
            elapsed_ns = %self.started.elapsed().as_nanos());
    }
}

pub(super) fn select_all(handle: usize, observations: &mut Observations) {
    let focus = recipient(handle, observations);
    let mut process = 0;
    let (thread, error) = observations.operation("GetWindowThreadProcessId", || {
        let thread = unsafe { GetWindowThreadProcessId(focus, Some(&mut process)) };
        (thread, (thread == 0).then(std::io::Error::last_os_error))
    });
    assert_ne!(
        thread, 0,
        "actual recipient thread lookup failed: {error:?}"
    );
    let layout =
        observations.operation("GetKeyboardLayout", || unsafe { GetKeyboardLayout(thread) });
    assert!(!layout.is_invalid(), "actual recipient HKL unavailable");
    observations.record(
        json!({"kind":"mapping_recipient", "window":format!("{focus:?}"),
        "thread_id":thread,"process_id":process,"hkl":format!("{layout:?}")}),
    );
    let mut records = Vec::new();
    for (key, released) in [(0x11, false), (0x41, false), (0x41, true), (0x11, true)] {
        let mapped = observations.operation("MapVirtualKeyExW", || unsafe {
            MapVirtualKeyExW(key, MAPVK_VK_TO_VSC_EX, Some(layout))
        });
        assert_ne!(mapped & 0xff, 0);
        assert!(matches!(mapped >> 8, 0 | 0xe0));
        observations.record(
            json!({"kind":"mapped_key", "virtual_key":key, "released":released,"mapped":mapped}),
        );
        records.push(record(
            u16::try_from(mapped & 0xff).expect("scan byte"),
            KEYEVENTF_SCANCODE
                | if mapped >> 8 == 0xe0 {
                    KEYEVENTF_EXTENDEDKEY
                } else {
                    KEYBD_EVENT_FLAGS::default()
                }
                | if released {
                    KEYEVENTF_KEYUP
                } else {
                    KEYBD_EVENT_FLAGS::default()
                },
        ));
    }
    deliver(&records, &[(0, 3), (1, 2)], observations);
}

fn recipient(handle: usize, observations: &mut Observations) -> windows::Win32::Foundation::HWND {
    let mut info = GUITHREADINFO {
        cbSize: size_of::<GUITHREADINFO>() as u32,
        ..Default::default()
    };
    observations
        .operation("GetGUIThreadInfo", || unsafe {
            GetGUIThreadInfo(0, &mut info)
        })
        .expect("actual foreground GUI queue");
    assert!(!info.hwndFocus.is_invalid(), "actual foreground recipient");
    let host = windows::Win32::Foundation::HWND(handle as *mut _);
    let foreground = observations.operation("GetForegroundWindow", || unsafe {
        windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow()
    });
    let owned = info.hwndFocus == host
        || observations
            .operation("IsChild", || unsafe {
                windows::Win32::UI::WindowsAndMessaging::IsChild(host, info.hwndFocus)
            })
            .as_bool();
    observations.record(
        json!({"kind":"native_recipient", "host":format!("{host:?}"),
        "foreground":format!("{foreground:?}"),"active":format!("{:?}",info.hwndActive),
        "focus":format!("{:?}",info.hwndFocus),"within_host":owned}),
    );
    assert_eq!(foreground, host);
    assert!(owned, "actual recipient belongs to the owned host");
    info.hwndFocus
}

pub(super) fn insert_unicode(handle: usize, value: &str, observations: &mut Observations) {
    let _ = recipient(handle, observations);
    observations
        .record(json!({"kind":"unicode_input", "units":value.encode_utf16().collect::<Vec<_>>()}));
    let mut records = Vec::new();
    for unit in value.encode_utf16() {
        records.push(record(unit, KEYEVENTF_UNICODE));
        records.push(record(unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
    }
    let pairs: Vec<_> = (0..records.len()).step_by(2).map(|i| (i, i + 1)).collect();
    deliver(&records, &pairs, observations);
}

fn deliver(records: &[INPUT], pairs: &[(usize, usize)], observations: &mut Observations) {
    let (sent, error) = observations.operation("SendInput", || {
        let sent = unsafe { SendInput(records, size_of::<INPUT>() as i32) } as usize;
        (
            sent,
            (sent != records.len()).then(std::io::Error::last_os_error),
        )
    });
    observations.record(
        json!({"kind":"input_admission", "requested":records.len(),"admitted":sent,
        "native_error":error.as_ref().map(std::io::Error::raw_os_error)}),
    );
    if sent != records.len() {
        let primary = error.expect("immediately captured native error");
        for &(down, up) in pairs {
            if sent > down && sent <= up {
                let (released, error) = observations.operation("SendInput_release", || {
                    let released =
                        unsafe { SendInput(&records[up..=up], size_of::<INPUT>() as i32) };
                    (
                        released,
                        (released != 1).then(std::io::Error::last_os_error),
                    )
                });
                observations.record(
                    json!({"kind":"release_admission", "down":down,"up":up,"admitted":released,
                    "native_error":error.as_ref().map(std::io::Error::raw_os_error)}),
                );
                if released != 1 {
                    let error = error.expect("immediately captured cleanup error");
                    tracing::error!(down, up, %error, "native unmatched-key release failed");
                }
            }
        }
        panic!(
            "SendInput admitted {sent}/{}; raw native error: {primary}",
            records.len()
        );
    }
    assert_eq!(sent, records.len(), "actual OS input insertion count");
}

fn record(scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wScan: scan,
                dwFlags: flags,
                ..Default::default()
            },
        },
    }
}
