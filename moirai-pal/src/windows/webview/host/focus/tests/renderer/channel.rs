//! One opt-in point observation; stdout retains its original completion bytes.

use super::super::{ObservationMode, client, input::Observations};
use crate::webview_limits::{MAX_WEBVIEW_MESSAGE_BYTES, MAX_WEBVIEW_MESSAGE_UNITS};
use std::{
    io::{self, Read, Write},
    os::windows::io::AsRawHandle,
    process::ChildStdout,
    time::Instant,
};
use windows::Win32::{
    Foundation::{ERROR_BROKEN_PIPE, HANDLE},
    System::Pipes::PeekNamedPipe,
};

const PREFIX: &str = "MOIRAI_UIA_MISMATCH ";
const ENVIRONMENT: &str = "MOIRAI_NATIVE_UIA_RENDERER_PROBE";

pub(super) fn enabled() -> bool {
    std::env::var_os(ENVIRONMENT).is_some_and(|value| {
        assert_eq!(value, "1", "explicit opt-in renderer probe");
        true
    })
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Progress {
    schema: u8,
    window: String,
    mode: String,
    replacement: usize,
    field: String,
    expected: Vec<u16>,
    actual: Vec<u16>,
    queries: usize,
}

impl Progress {
    pub(super) fn validate(&self, handle: usize, mode: ObservationMode) -> io::Result<()> {
        let ordinal = self
            .replacement
            .checked_sub(1)
            .filter(|ordinal| *ordinal < client::CASES.len() * client::FIELDS.len())
            .ok_or_else(|| io::Error::other("probe replacement outside the actual input domain"))?;
        let row = client::CASES[ordinal / client::FIELDS.len()];
        let column = ordinal % client::FIELDS.len();
        let expected: Vec<_> = [row.0, row.1, row.2, row.3][column]
            .encode_utf16()
            .collect();
        if self.schema != 1
            || self.window != handle.to_string()
            || self.mode != mode.as_str()
            || self.field != client::FIELDS[column]
            || self.expected != expected
            || self.actual == self.expected
            || self.queries == 0
            || self.actual.len() > MAX_WEBVIEW_MESSAGE_UNITS
        {
            return Err(io::Error::other(
                "probe progress does not bind actual native mismatch",
            ));
        }
        Ok(())
    }
}

pub(in super::super) struct Emitter {
    enabled: bool,
    emitted: bool,
    handle: usize,
    mode: ObservationMode,
}

impl Emitter {
    pub(in super::super) fn new(handle: usize, mode: ObservationMode) -> Self {
        Self {
            enabled: enabled(),
            emitted: false,
            handle,
            mode,
        }
    }

    pub(in super::super) fn mismatch(
        &mut self,
        observations: &Observations,
        expected: &[u16],
        actual: &[u16],
        queries: usize,
        deadline: Instant,
    ) {
        if !self.enabled || self.emitted || !observations.exhausted() || Instant::now() >= deadline
        {
            return;
        }
        // Mark before emission: a transport error never retries the probe.
        self.emitted = true;
        let progress = Progress {
            schema: 1,
            window: self.handle.to_string(),
            mode: self.mode.as_str().into(),
            replacement: observations.replacement(),
            field: observations.field_name().into(),
            expected: expected.to_owned(),
            actual: actual.to_owned(),
            queries,
        };
        progress
            .validate(self.handle, self.mode)
            .expect("actual mismatch domain");
        let encoded = serde_json::to_string(&progress).expect("native progress JSON");
        assert!(encoded.len() + PREFIX.len() + 2 <= MAX_WEBVIEW_MESSAGE_BYTES);
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "\n{PREFIX}{encoded}").expect("native progress output");
        stdout.flush().expect("native progress flush");
    }
}

#[derive(Default)]
pub(super) struct Channel {
    pub(super) bytes: Vec<u8>,
    scanned: usize,
    seen: bool,
}

impl Channel {
    pub(super) fn poll(&mut self, stdout: &mut ChildStdout) -> io::Result<Option<Progress>> {
        let mut available = 0;
        // SAFETY: ChildStdout owns this readable pipe handle. Peek copies only
        // the count; it neither waits for a writer nor consumes partial UTF-8.
        let result = unsafe {
            PeekNamedPipe(
                HANDLE(stdout.as_raw_handle()),
                None,
                0,
                None,
                Some(&mut available),
                None,
            )
        };
        match result {
            Ok(()) => {}
            Err(error) if error.code() == ERROR_BROKEN_PIPE.to_hresult() => return Ok(None),
            Err(error) => return Err(io::Error::other(error)),
        }
        if available > 0 {
            let mut chunk = [0; 4096];
            let room = MAX_WEBVIEW_MESSAGE_BYTES.saturating_sub(self.bytes.len()) + 1;
            let length = usize::try_from(available)
                .map_err(io::Error::other)?
                .min(chunk.len())
                .min(room);
            let read = stdout.read(&mut chunk[..length])?;
            return self.append(&chunk[..read]);
        }
        Ok(None)
    }

    fn append(&mut self, bytes: &[u8]) -> io::Result<Option<Progress>> {
        if bytes.len() > MAX_WEBVIEW_MESSAGE_BYTES.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other(
                "UIA client output exceeds the existing byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        let mut progress = None;
        while let Some(end) = self.bytes[self.scanned..]
            .iter()
            .position(|byte| *byte == b'\n')
        {
            let end = self.scanned + end;
            let line =
                std::str::from_utf8(&self.bytes[self.scanned..end]).map_err(io::Error::other)?;
            if let Some(encoded) = line.strip_prefix(PREFIX) {
                if self.seen {
                    return Err(io::Error::other("duplicate native renderer probe progress"));
                }
                self.seen = true;
                progress = Some(
                    serde_json::from_str(encoded.trim_end_matches('\r'))
                        .map_err(io::Error::other)?,
                );
            }
            self.scanned = end + 1;
        }
        Ok(progress)
    }
}

pub(super) fn validate_terminal(
    bytes: &[u8],
    handle: usize,
    mode: ObservationMode,
) -> io::Result<bool> {
    let mut channel = Channel::default();
    let progress = channel.append(bytes)?;
    if let Some(progress) = &progress {
        progress.validate(handle, mode)?;
    }
    if channel.scanned != bytes.len() {
        return Err(io::Error::other(
            "UIA client output ended with an incomplete line",
        ));
    }
    Ok(progress.is_some())
}

#[cfg(test)]
mod tests;
