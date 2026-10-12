//! One monotonic deadline covers a complete Unix instance message.
//!
//! Darwin rejects SO_RCVTIMEO after a sender closes, including a socket that
//! still holds a complete buffered message. Nonblocking reads plus poll admit
//! buffered bytes and preserve the message deadline without changing socket
//! timeout state after peer shutdown.

use super::INSTANCE_IO_TIMEOUT;
use std::{
    io::{self, ErrorKind, Read},
    os::{fd::AsRawFd, unix::net::UnixStream},
    time::Instant,
};

pub(super) struct Receiver<'a> {
    stream: &'a mut UnixStream,
    deadline: Instant,
}

impl<'a> Receiver<'a> {
    pub(super) fn new(stream: &'a mut UnixStream) -> Self {
        Self {
            stream,
            deadline: Instant::now() + INSTANCE_IO_TIMEOUT,
        }
    }

    fn wait(&self) -> io::Result<()> {
        loop {
            let remaining = self.deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(stalled());
            }
            // poll counts whole milliseconds; ceiling preserves a positive
            // sub-millisecond remainder. The fixed message deadline bounds
            // this conversion to INSTANCE_IO_TIMEOUT in milliseconds.
            let millis = remaining.as_millis()
                + u128::from(!remaining.subsec_nanos().is_multiple_of(1_000_000));
            let timeout = i32::try_from(millis).map_err(io::Error::other)?;
            let mut descriptor = libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: descriptor is one initialized pollfd; its owned stream
            // remains borrowed and open for this bounded call.
            let result = unsafe { libc::poll(&mut descriptor, 1, timeout) };
            match result {
                0 => return Err(stalled()),
                -1 => {
                    let error = io::Error::last_os_error();
                    if error.kind() != ErrorKind::Interrupted {
                        return Err(error);
                    }
                }
                _ => return Ok(()),
            }
        }
    }
}

impl Read for Receiver<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            if Instant::now() >= self.deadline {
                return Err(stalled());
            }
            match self.stream.read(buffer) {
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) if error.kind() == ErrorKind::WouldBlock => self.wait()?,
                result => return result,
            }
        }
    }
}

fn stalled() -> io::Error {
    io::Error::new(ErrorKind::TimedOut, "instance sender stalled")
}

#[cfg(test)]
mod tests;
