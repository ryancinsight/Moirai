//! Independent bounded length-prefix framing over raw Quinn streams.

use super::error::{Error, Result};
use super::material::OPERATION_TIMEOUT;

pub(super) struct Stream {
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    limit: u32,
    send_finished: bool,
}

impl Stream {
    pub(super) const fn new(send: quinn::SendStream, recv: quinn::RecvStream, limit: u32) -> Self {
        Self {
            send,
            recv,
            limit,
            send_finished: false,
        }
    }

    pub(super) fn id(&self) -> u64 {
        u64::from(self.send.id())
    }

    pub(super) async fn send_message(&mut self, bytes: &[u8]) -> Result<()> {
        let size = u64::try_from(bytes.len()).map_err(Error::Numeric)?;
        if size > u64::from(self.limit) {
            return Err(Error::MessageTooLarge {
                size,
                limit: self.limit,
            });
        }
        let length = u32::try_from(bytes.len()).map_err(Error::Numeric)?;
        let result = tokio::time::timeout(OPERATION_TIMEOUT, async {
            self.send
                .write_all(&length.to_be_bytes())
                .await
                .map_err(Error::Write)?;
            self.send.write_all(bytes).await.map_err(Error::Write)
        })
        .await
        .map_err(|_| Error::TimedOut("send message"))?;
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub(super) async fn recv_message(&mut self) -> Result<Option<Vec<u8>>> {
        let result = tokio::time::timeout(OPERATION_TIMEOUT, async {
            let mut prefix = [0; 4];
            let first = self
                .recv
                .read(&mut prefix[..1])
                .await
                .map_err(Error::Read)?;
            let Some(first) = first else {
                return Ok(None);
            };
            if first != 1 {
                return Err(Error::Closed);
            }
            self.recv
                .read_exact(&mut prefix[1..])
                .await
                .map_err(Error::ReadExact)?;
            let length = u32::from_be_bytes(prefix);
            if length > self.limit {
                return Err(Error::MessageTooLarge {
                    size: u64::from(length),
                    limit: self.limit,
                });
            }
            let capacity = usize::try_from(length).map_err(Error::Numeric)?;
            let mut payload = Vec::new();
            payload
                .try_reserve_exact(capacity)
                .map_err(Error::Allocation)?;
            payload.resize(capacity, 0);
            self.recv
                .read_exact(&mut payload)
                .await
                .map_err(Error::ReadExact)?;
            Ok(Some(payload))
        })
        .await
        .map_err(|_| Error::TimedOut("receive message"))?;
        if result.is_err() {
            self.abort();
        }
        result
    }

    pub(super) async fn finish_and_wait(&mut self) -> Result<()> {
        if !self.send_finished {
            self.finish()?;
        }
        let stopped = tokio::time::timeout(OPERATION_TIMEOUT, self.send.stopped())
            .await
            .map_err(|_| Error::TimedOut("finish stream"))?
            .map_err(Error::Stop)?;
        if let Some(code) = stopped {
            return Err(Error::PeerStopped(code));
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> Result<()> {
        self.send.finish().map_err(|_| Error::Closed)?;
        self.send_finished = true;
        Ok(())
    }

    fn abort(&mut self) {
        if let Err(error) = self.send.reset(quinn::VarInt::from_u32(1)) {
            tracing::debug!(%error, "reference send half is already closed during abort");
        }
        if let Err(error) = self.recv.stop(quinn::VarInt::from_u32(1)) {
            tracing::debug!(%error, "reference receive half is already closed during abort");
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.send_finished
            && let Err(error) = self.send.reset(quinn::VarInt::from_u32(1))
        {
            tracing::debug!(%error, "reference send half is already closed during drop");
        }
        if let Err(error) = self.recv.stop(quinn::VarInt::from_u32(1)) {
            tracing::debug!(%error, "reference receive half is already closed during drop");
        }
    }
}
