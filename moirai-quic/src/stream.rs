use super::error::bounded;
use super::{QuicConfig, QuicError};

/// One ordered, reliable application-message stream.
///
/// Wire format: four-byte big-endian length, then exactly that many bytes.
/// A cancelled or failed I/O future aborts this stream, so a partial message is
/// never interpreted as a new prefix. Sibling streams remain independent.
pub struct QuicStream {
    pub(super) send: moirai_quinn::SendStream,
    pub(super) recv: moirai_quinn::RecvStream,
    max_frame_bytes: u32,
    deadline: std::time::Duration,
    alpn: std::sync::Arc<[u8]>,
    health: std::sync::Arc<super::runtime::RuntimeHealth>,
    _permit: super::admission::StreamPermit,
    aborted: bool,
    send_finished: bool,
    receive_finished: bool,
}

impl QuicStream {
    pub(super) fn new(
        (send, recv): (moirai_quinn::SendStream, moirai_quinn::RecvStream),
        config: &QuicConfig,
        health: std::sync::Arc<super::runtime::RuntimeHealth>,
        permit: super::admission::StreamPermit,
    ) -> Self {
        Self {
            send,
            recv,
            max_frame_bytes: config.max_frame_bytes,
            deadline: config.operation_timeout,
            alpn: std::sync::Arc::clone(&config.alpn),
            health,
            _permit: permit,
            aborted: false,
            send_finished: false,
            receive_finished: false,
        }
    }

    /// Maximum application bytes in one message.
    pub const fn max_frame_bytes(&self) -> u32 {
        self.max_frame_bytes
    }

    /// TLS-negotiated application protocol of this stream's parent connection.
    pub fn alpn(&self) -> &[u8] {
        &self.alpn
    }

    /// QUIC stream identifier, including initiator and direction bits.
    pub fn id(&self) -> u64 {
        u64::from(self.send.id())
    }

    /// Write one complete bounded message within one deadline.
    ///
    /// # Errors
    /// Returns size, closed, timeout or write failures. Oversized local input is
    /// rejected before writing and leaves framing intact; failed/partial writes
    /// invalidate the stream.
    pub async fn send_message(&mut self, bytes: &[u8]) -> Result<(), QuicError> {
        self.health.check()?;
        let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if size > u64::from(self.max_frame_bytes) {
            return Err(QuicError::MessageTooLarge {
                size,
                limit: self.max_frame_bytes,
            });
        }
        if self.aborted || self.send_finished {
            return Err(QuicError::Closed);
        }
        let deadline = self.deadline;
        let mut guard = FrameGuard {
            stream: self,
            complete: false,
        };
        let operation = async {
            let length = u32::try_from(bytes.len()).map_err(|_| QuicError::MessageTooLarge {
                size,
                limit: guard.stream.max_frame_bytes,
            })?;
            guard
                .stream
                .send
                .write_all(&length.to_be_bytes())
                .await
                .map_err(QuicError::Write)?;
            guard
                .stream
                .send
                .write_all(bytes)
                .await
                .map_err(QuicError::Write)
        };
        let result = bounded(deadline, "send message", operation).await;
        guard.stream.health.check()?;
        result??;
        guard.complete = true;
        Ok(())
    }

    /// Read one complete message. `None` means clean EOF before a new prefix.
    ///
    /// Length admission precedes payload allocation. Truncation, oversized
    /// prefixes, timeout and cancellation abort the stream. Empty messages are
    /// valid and distinct from EOF.
    ///
    /// # Errors
    /// Returns size, allocation, timeout, truncation, reset or connection errors.
    pub async fn recv_message(&mut self) -> Result<Option<Vec<u8>>, QuicError> {
        self.health.check()?;
        if self.aborted {
            return Err(QuicError::Closed);
        }
        if self.receive_finished {
            return Ok(None);
        }
        let deadline = self.deadline;
        let mut guard = FrameGuard {
            stream: self,
            complete: false,
        };
        let operation = async {
            let mut prefix = [0; 4];
            if guard
                .stream
                .recv
                .read(&mut prefix[..1])
                .await
                .map_err(QuicError::Read)?
                .is_none()
            {
                guard.stream.receive_finished = true;
                return Ok(None);
            }
            guard
                .stream
                .recv
                .read_exact(&mut prefix[1..])
                .await
                .map_err(QuicError::ReadExact)?;
            let length = u32::from_be_bytes(prefix);
            if length > guard.stream.max_frame_bytes {
                return Err(QuicError::MessageTooLarge {
                    size: u64::from(length),
                    limit: guard.stream.max_frame_bytes,
                });
            }
            let capacity = usize::try_from(length).map_err(|_| QuicError::MessageTooLarge {
                size: u64::from(length),
                limit: guard.stream.max_frame_bytes,
            })?;
            let mut payload = Vec::new();
            payload.try_reserve_exact(capacity).map_err(|error| {
                QuicError::Io(std::io::Error::new(std::io::ErrorKind::OutOfMemory, error))
            })?;
            payload.resize(capacity, 0);
            guard
                .stream
                .recv
                .read_exact(&mut payload)
                .await
                .map_err(QuicError::ReadExact)?;
            Ok(Some(payload))
        };
        let message = bounded(deadline, "receive message", operation).await;
        guard.stream.health.check()?;
        let message = message??;
        guard.complete = true;
        Ok(message)
    }

    /// Finish the sending half; the peer can still send its response.
    ///
    /// This queues FIN. Keep the parent connection alive until the peer consumes
    /// the response; closing the connection can discard outstanding data.
    ///
    /// # Errors
    /// Returns closed stream when sending is already finished or cancelled.
    pub fn finish(&mut self) -> Result<(), QuicError> {
        self.health.check()?;
        if self.aborted || self.send_finished {
            return Err(QuicError::Closed);
        }
        self.send.finish().map_err(|_| QuicError::Closed)?;
        self.send_finished = true;
        Ok(())
    }

    /// Queue FIN and wait for acknowledgement of every queued byte and FIN.
    ///
    /// This completes transport delivery before the connection owner closes.
    /// It does not prove application dispatch or authorize a command retry.
    /// Cancellation aborts the stream; peer STOP remains a distinct write error.
    ///
    /// # Errors
    /// Returns closed, deadline, peer-stop, connection or executor errors.
    pub async fn finish_and_wait(&mut self) -> Result<(), QuicError> {
        self.health.check()?;
        if self.aborted {
            return Err(QuicError::Closed);
        }
        if !self.send_finished {
            self.finish()?;
        }
        let deadline = self.deadline;
        let mut guard = FrameGuard {
            stream: self,
            complete: false,
        };
        let result = bounded(deadline, "finish stream", guard.stream.send.stopped()).await;
        guard.stream.health.check()?;
        let stopped = result?.map_err(|error| QuicError::Write(error.into()))?;
        if let Some(code) = stopped {
            return Err(QuicError::Write(moirai_quinn::WriteError::Stopped(code)));
        }
        guard.complete = true;
        Ok(())
    }

    /// Reset sending and stop receiving without closing sibling streams.
    ///
    /// Cleanup is best effort when the peer already closed the stream.
    pub fn abort(&mut self) {
        self.aborted = true;
        let _ = self.send.reset(super::error::CANCELLED);
        let _ = self.recv.stop(super::error::CANCELLED);
    }
}

struct FrameGuard<'a> {
    stream: &'a mut QuicStream,
    complete: bool,
}

impl Drop for FrameGuard<'_> {
    fn drop(&mut self) {
        if !self.complete {
            self.stream.abort();
        }
    }
}

impl Drop for QuicStream {
    fn drop(&mut self) {
        // Successful FIN preserves buffered bytes. A stream dropped without
        // FIN is an abandoned session, so partial output must not be delivered.
        if !self.send_finished {
            let _ = self.send.reset(super::error::CANCELLED);
        }
        let _ = self.recv.stop(super::error::CANCELLED);
    }
}
