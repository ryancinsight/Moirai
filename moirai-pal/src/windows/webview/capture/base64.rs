//! Strict canonical base64 for a bounded rendered-frame payload.
//!
//! For a nonempty canonical string of `q` quartets and terminal padding
//! `p ∈ {0, 1, 2}`, the decoded length is `L = 3q - p`. Each full quartet
//! contributes three bytes; one terminal `=` removes one byte and `==`
//! removes two. Checking `L <= LIMIT` before reservation makes the byte
//! budget inclusive without charging the absent padded bytes.
//!
//! The grammar requires the RFC 4648 alphabet, padding only in the terminal
//! quartet, and zero discarded bits. Before the terminal quartet the loop
//! produces at most `3(q - 1) <= L` bytes; terminal grammar validation occurs
//! before any terminal byte is appended. Thus malformed input cannot force
//! the vector to grow beyond the admitted exact output length.

use std::io;

pub(crate) fn decode<const LIMIT: usize>(data: &str) -> io::Result<Vec<u8>> {
    if data.is_empty() || !data.len().is_multiple_of(4) {
        return Err(invalid("WebView2 presented-frame data is not base64"));
    }
    let bytes = data.as_bytes();
    let padding = if bytes[bytes.len() - 1] == b'=' {
        1 + usize::from(bytes[bytes.len() - 2] == b'=')
    } else {
        0
    };
    let capacity = (bytes.len() / 4)
        .checked_mul(3)
        .and_then(|length| length.checked_sub(padding))
        .ok_or_else(|| invalid("WebView2 presented-frame data is too large"))?;
    if capacity > LIMIT {
        return Err(invalid(
            "WebView2 presented-frame data exceeds the bounded image budget",
        ));
    }
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(capacity)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "presented-frame allocation"))?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let last = index == bytes.len() / 4 - 1;
        let first = base64_value(chunk[0]).ok_or_else(|| invalid("invalid base64"))?;
        let second = base64_value(chunk[1]).ok_or_else(|| invalid("invalid base64"))?;
        let third = match chunk[2] {
            b'=' => None,
            value => Some(base64_value(value).ok_or_else(|| invalid("invalid base64"))?),
        };
        let fourth = match chunk[3] {
            b'=' => None,
            value => Some(base64_value(value).ok_or_else(|| invalid("invalid base64"))?),
        };
        if (third.is_none() && fourth.is_some()) || (!last && (third.is_none() || fourth.is_none()))
        {
            return Err(invalid("invalid base64 padding"));
        }
        let third = third.unwrap_or(0);
        let fourth = fourth.unwrap_or(0);
        if (chunk[2] == b'=' && second & 0x0f != 0) || (chunk[3] == b'=' && third & 0x03 != 0) {
            return Err(invalid("nonzero base64 padding bits"));
        }
        decoded.push(first << 2 | second >> 4);
        if chunk[2] != b'=' {
            decoded.push(second << 4 | third >> 2);
        }
        if chunk[3] != b'=' {
            decoded.push(third << 6 | fourth);
        }
    }
    Ok(decoded)
}

fn base64_value(value: u8) -> Option<u8> {
    match value {
        b'A'..=b'Z' => Some(value - b'A'),
        b'a'..=b'z' => Some(value - b'a' + 26),
        b'0'..=b'9' => Some(value - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "base64/tests.rs"]
mod tests;
