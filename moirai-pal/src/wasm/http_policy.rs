//! Allocation and deadline admission for browser HTTP.
use std::{io, time::Duration};
pub(crate) const MAX_HTTP_BYTES: usize = crate::file_policy::MAX_READ_BYTES;
pub(crate) const MAX_HTTP_DEADLINE: Duration = Duration::from_secs(60);

/// Fetch's bodyless statuses carry zero received body bytes. In particular,
/// a 304 Content-Length describes the selected representation (RFC 9110 §8.6).
/// https://fetch.spec.whatwg.org/#null-body-status
pub(crate) const fn response_has_body(status: u16) -> bool {
    !matches!(status, 101 | 103 | 204 | 205 | 304)
}

pub(crate) fn validate_limits(body: usize, bound: usize, deadline: Duration) -> io::Result<usize> {
    if body > MAX_HTTP_BYTES
        || bound == 0
        || bound > MAX_HTTP_BYTES
        || deadline.is_zero()
        || deadline > MAX_HTTP_DEADLINE
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Browser HTTP byte budget or deadline is outside provider limits",
        ));
    }
    Ok(bound)
}

pub(crate) fn content_length(value: &str, bound: usize) -> io::Result<usize> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Browser HTTP Content-Length is not decimal",
        ));
    }
    let length = value.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Browser HTTP Content-Length overflows",
        )
    })?;
    count_chunk(0, length, bound)
}

/// b + c <= B iff c <= B - b, after validating b <= B.
pub(crate) fn count_chunk(count: usize, chunk: usize, bound: usize) -> io::Result<usize> {
    if count > bound || chunk > bound - count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Browser byte stream exceeds its read bound",
        ));
    }
    Ok(count + chunk)
}

#[cfg(test)]
#[path = "http_policy/tests.rs"]
mod tests;
