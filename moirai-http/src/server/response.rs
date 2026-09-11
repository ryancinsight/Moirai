//! Validated HTTP response framing for the one-shot server.

use std::io;

/// A validated HTTP response owned by the application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub(super) status: u16,
    pub(super) headers: Vec<(String, String)>,
    pub(super) body: Vec<u8>,
}

impl HttpResponse {
    /// Construct a response with a status and body.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when `status` is not a final
    /// three-digit HTTP status code.
    pub fn new<B: Into<Vec<u8>>>(status: u16, body: B) -> io::Result<Self> {
        if !(200..=599).contains(&status) {
            return Err(invalid_input("HTTP response status must be three digits"));
        }
        Ok(Self {
            status,
            headers: Vec::new(),
            body: body.into(),
        })
    }

    /// Add one application response header.
    ///
    /// `Content-Length`, `Connection`, and `Transfer-Encoding` are transport
    /// owned and cannot be overridden.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] for an invalid, reserved, or
    /// duplicate header.
    pub fn set_header<N: Into<String>, V: Into<String>>(
        &mut self,
        name: N,
        value: V,
    ) -> io::Result<()> {
        let name = name.into();
        let value = value.into();
        validate_header_name(&name)?;
        validate_header_value(&value)?;
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "content-length" | "connection" | "transfer-encoding"
        ) {
            return Err(invalid_input(
                "HTTP response framing header is transport-owned",
            ));
        }
        if self
            .headers
            .iter()
            .any(|(existing, _)| existing.eq_ignore_ascii_case(&name))
        {
            return Err(invalid_input("duplicate HTTP response header"));
        }
        self.headers.push((name, value));
        Ok(())
    }

    /// Return the response status code.
    #[must_use]
    pub const fn status(&self) -> u16 {
        self.status
    }

    /// Return response headers in insertion order.
    #[must_use]
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// Return the response body.
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

pub(super) fn encode_response_head(
    response: &HttpResponse,
    max_response_bytes: usize,
) -> io::Result<Vec<u8>> {
    let status = response.status;
    let reason = reason_phrase(status);
    let mut head = Vec::new();
    let status_line = format!("HTTP/1.1 {status} {reason}\r\n");
    head.try_reserve(status_line.len())
        .map_err(|_| invalid_data("HTTP response header allocation exceeds available memory"))?;
    head.extend_from_slice(status_line.as_bytes());
    for (name, value) in &response.headers {
        let line_length = name
            .len()
            .checked_add(2)
            .and_then(|length| length.checked_add(value.len()))
            .and_then(|length| length.checked_add(2))
            .ok_or_else(|| invalid_data("HTTP response header size overflows its bound"))?;
        let projected = head
            .len()
            .checked_add(line_length)
            .ok_or_else(|| invalid_data("HTTP response header size overflows its bound"))?;
        if projected > max_response_bytes {
            return Err(invalid_data("HTTP response exceeds configured byte bound"));
        }
        head.extend_from_slice(name.as_bytes());
        head.extend_from_slice(b": ");
        head.extend_from_slice(value.as_bytes());
        head.extend_from_slice(b"\r\n");
    }
    let framing = format!(
        "Content-Length: {}\r\nConnection: close\r\n\r\n",
        response.body.len()
    );
    let projected = head
        .len()
        .checked_add(framing.len())
        .ok_or_else(|| invalid_data("HTTP response header size overflows its bound"))?;
    if projected > max_response_bytes {
        return Err(invalid_data("HTTP response exceeds configured byte bound"));
    }
    head.extend_from_slice(framing.as_bytes());
    Ok(head)
}

fn validate_header_name(name: &str) -> io::Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
    {
        return Err(invalid_input("HTTP response header name is not a token"));
    }
    Ok(())
}

fn validate_header_value(value: &str) -> io::Result<()> {
    if value
        .bytes()
        .any(|byte| byte < 0x20 && byte != b'\t' || byte == 0x7f)
    {
        return Err(invalid_input(
            "HTTP response header value contains a control byte",
        ));
    }
    Ok(())
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        206 => "Partial Content",
        300 => "Multiple Choices",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_rejects_transport_headers_and_controls() {
        let mut response = HttpResponse::new(200, b"ok".to_vec()).expect("valid response");
        assert!(response.set_header("Content-Length", "2").is_err());
        assert!(
            response
                .set_header("X-Test", "ok\r\nInjected: true")
                .is_err()
        );
        response.set_header("X-Test", "ok").expect("valid header");
        assert!(response.set_header("x-test", "again").is_err());
    }
}
