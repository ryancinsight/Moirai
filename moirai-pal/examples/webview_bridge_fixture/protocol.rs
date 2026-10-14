//! One wire owner shared by the native fixture and its compiled Rust payload.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Command {
    Echo {
        mount: u8,
        identity: String,
        patient: String,
        values: [u32; 3],
    },
    Boundary {
        mount: u8,
        text: String,
    },
    Probe {
        mount: u8,
        identity: String,
    },
    Close {
        mount: u8,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Notice {
    Ready {
        mount: u8,
        old_calls: u32,
    },
    Echo {
        mount: u8,
        identity: String,
        patient: String,
        total: u32,
        calls: u32,
        old_calls: u32,
    },
    Boundary {
        mount: u8,
        text: String,
    },
    Rejected {
        mount: u8,
        bytes: usize,
        error_kind: String,
        error: String,
    },
    Probe {
        mount: u8,
        identity: String,
        calls: u32,
        old_calls: u32,
    },
    Closed {
        mount: u8,
        calls: u32,
        old_calls: u32,
        retained: usize,
    },
}
