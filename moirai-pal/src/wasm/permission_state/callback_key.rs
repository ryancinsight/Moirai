//! Realm-wide callback identities, independent of per-document generations.

use std::io;

/// Nonzero identities never wrap or reset while a browser realm remains alive.
/// Decimal-string binding preserves all bits through the JavaScript boundary.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CallbackKey(u64);

impl CallbackKey {
    pub(crate) fn encode(self) -> String {
        self.0.to_string()
    }

    pub(crate) fn decode(value: &str) -> Option<Self> {
        let bytes = value.as_bytes();
        if !matches!(bytes.first(), Some(b'1'..=b'9')) || !bytes.iter().all(u8::is_ascii_digit) {
            return None;
        }
        value.parse().ok().map(Self)
    }
}

#[derive(Debug, Default)]
pub(crate) struct CallbackSequence(u64);

impl CallbackSequence {
    pub(crate) fn next(&mut self) -> io::Result<CallbackKey> {
        let next = self.0.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Geolocation callback identity exhausted",
            )
        })?;
        self.0 = next;
        Ok(CallbackKey(next))
    }
}

#[cfg(test)]
#[path = "callback_key/tests.rs"]
mod tests;
