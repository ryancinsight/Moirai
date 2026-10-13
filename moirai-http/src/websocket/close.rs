//! Validated peer-close observations and WebSocket protocol errors.

use std::fmt;
use std::io;

/// A complete, validated RFC 6455 close frame received from the peer.
///
/// [`super::WebSocketStream::recv_message`] retains `UnexpectedEof` as its I/O
/// category and stores this typed cause in [`io::Error::get_ref`]. Truncated
/// frames never carry this cause. An empty close payload has no status code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSocketPeerClosed {
    code: Option<u16>,
    reason: String,
}

impl WebSocketPeerClosed {
    /// Validated wire status, absent when the close payload is empty.
    pub const fn code(&self) -> Option<u16> {
        self.code
    }

    /// Validated UTF-8 close reason from the peer.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for WebSocketPeerClosed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Retain the established diagnostic; structured fields carry context.
        formatter.write_str("WebSocket peer closed the connection")
    }
}

impl std::error::Error for WebSocketPeerClosed {}

// A borrowed validated payload avoids allocation for locally sent close frames.
// Only this leaf constructs it, so conversion cannot fabricate validated fields.
pub(super) struct ClosePayload<'a> {
    code: Option<u16>,
    reason: &'a str,
}

impl From<ClosePayload<'_>> for WebSocketPeerClosed {
    fn from(payload: ClosePayload<'_>) -> Self {
        Self {
            code: payload.code,
            // io::Error owns a 'static cause after its frame buffer is released.
            reason: payload.reason.to_owned(),
        }
    }
}

pub(super) fn parse_close_payload(payload: &[u8]) -> io::Result<ClosePayload<'_>> {
    match payload {
        [] => Ok(ClosePayload {
            code: None,
            reason: "",
        }),
        [_] => Err(protocol_error("WebSocket close payload has one byte")),
        [high, low, reason @ ..] => {
            let code = u16::from_be_bytes([*high, *low]);
            let valid_range = (1000..=2999).contains(&code) || (3000..=4999).contains(&code);
            if !valid_range || matches!(code, 1004 | 1005 | 1006 | 1015) {
                return Err(protocol_error("WebSocket close code is reserved"));
            }
            let reason = std::str::from_utf8(reason)
                .map_err(|_| protocol_error("WebSocket close reason is not UTF-8"))?;
            Ok(ClosePayload {
                code: Some(code),
                reason,
            })
        }
    }
}

pub(super) fn protocol_error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
