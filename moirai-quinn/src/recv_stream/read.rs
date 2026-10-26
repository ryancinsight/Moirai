use super::*;

impl RecvStream {
    pub(crate) fn new(conn: ConnectionRef, stream: StreamId, is_0rtt: bool) -> Self {
        Self {
            conn,
            stream,
            is_0rtt,
            all_data_read: false,
            reset: None,
        }
    }

    /// Read data contiguously from the stream.
    ///
    /// Yields the number of bytes read into `buf` on success, or `None` if the stream was finished.
    ///
    /// This operation is cancel-safe.
    pub async fn read(&mut self, buf: &mut [u8]) -> Result<Option<usize>, ReadError> {
        Read {
            stream: self,
            buf: ReadBuffer::new(buf),
        }
        .await
    }

    /// Read an exact number of bytes contiguously from the stream.
    ///
    /// See [`read()`] for details. This operation is *not* cancel-safe.
    ///
    /// [`read()`]: RecvStream::read
    pub async fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError> {
        ReadExact {
            stream: self,
            buf: ReadBuffer::new(buf),
        }
        .await
    }

    /// Attempts to read from the stream into the provided buffer
    ///
    /// On success, returns `Poll::Ready(Ok(num_bytes_read))` and places data into `buf`. If this
    /// returns zero bytes read (and `buf` has a non-zero length), that indicates that the remote
    /// side has [`finish`]ed the stream and the local side has already read all bytes.
    ///
    /// If no data is available for reading, this returns `Poll::Pending` and arranges for the
    /// current task (via `cx.waker()`) to be notified when the stream becomes readable or is
    /// closed.
    ///
    /// [`finish`]: crate::SendStream::finish
    pub fn poll_read(
        &mut self,
        cx: &mut Context,
        buf: &mut [u8],
    ) -> Poll<Result<usize, ReadError>> {
        let mut buf = ReadBuffer::new(buf);
        ready!(self.poll_read_buf(cx, &mut buf))?;
        Poll::Ready(Ok(buf.filled().len()))
    }

    /// Attempts to read from the stream into the provided buffer, which may be uninitialized
    ///
    /// On success, returns `Poll::Ready(Ok(()))` and places data into the unfilled portion of
    /// `buf`. If this does not write any bytes to `buf` (and `buf.remaining()` is non-zero), that
    /// indicates that the remote side has [`finish`]ed the stream and the local side has already
    /// read all bytes.
    ///
    /// If no data is available for reading, this returns `Poll::Pending` and arranges for the
    /// current task (via `cx.waker()`) to be notified when the stream becomes readable or is
    /// closed.
    ///
    /// [`finish`]: crate::SendStream::finish
    pub fn poll_read_buf(
        &mut self,
        cx: &mut Context,
        buf: &mut ReadBuffer<'_>,
    ) -> Poll<Result<(), ReadError>> {
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }

        self.poll_read_generic(cx, true, |chunks| {
            let mut read = false;
            loop {
                if buf.remaining() == 0 {
                    // We know `read` is `true` because `buf.remaining()` was not 0 before
                    return ReadStatus::Readable(());
                }

                match chunks.next(buf.remaining()) {
                    Ok(Some(chunk)) => {
                        buf.put_slice(&chunk.bytes);
                        read = true;
                    }
                    res => return (if read { Some(()) } else { None }, res.err()).into(),
                }
            }
        })
        .map(|res| res.map(|_| ()))
    }

    /// Read the next segment of data
    ///
    /// Yields `None` if the stream was finished. Otherwise, yields a segment of data and its
    /// offset in the stream. If `ordered` is `true`, the chunk's offset will be immediately after
    /// the last data yielded by `read()` or `read_chunk()`. If `ordered` is `false`, segments may
    /// be received in any order, and the `Chunk`'s `offset` field can be used to determine
    /// ordering in the caller. Unordered reads are less prone to head-of-line blocking within a
    /// stream, but require the application to manage reassembling the original data.
    ///
    /// Slightly more efficient than `read` due to not copying. Chunk boundaries do not correspond
    /// to peer writes, and hence cannot be used as framing.
    ///
    /// This operation is cancel-safe.
    pub async fn read_chunk(
        &mut self,
        max_length: usize,
        ordered: bool,
    ) -> Result<Option<Chunk>, ReadError> {
        ReadChunk {
            stream: self,
            max_length,
            ordered,
        }
        .await
    }

    /// Attempts to read a chunk from the stream.
    ///
    /// On success, returns `Poll::Ready(Ok(Some(chunk)))`. If `Poll::Ready(Ok(None))`
    /// is returned, it implies that EOF has been reached.
    ///
    /// If no data is available for reading, the method returns `Poll::Pending`
    /// and arranges for the current task (via cx.waker()) to receive a notification
    /// when the stream becomes readable or is closed.
    pub(super) fn poll_read_chunk(
        &mut self,
        cx: &mut Context,
        max_length: usize,
        ordered: bool,
    ) -> Poll<Result<Option<Chunk>, ReadError>> {
        self.poll_read_generic(cx, ordered, |chunks| match chunks.next(max_length) {
            Ok(Some(chunk)) => ReadStatus::Readable(chunk),
            res => (None, res.err()).into(),
        })
    }

    /// Read the next segments of data
    ///
    /// Fills `bufs` with the segments of data beginning immediately after the
    /// last data yielded by `read` or `read_chunk`, or `None` if the stream was
    /// finished.
    ///
    /// Slightly more efficient than `read` due to not copying. Chunk boundaries
    /// do not correspond to peer writes, and hence cannot be used as framing.
    ///
    /// This operation is cancel-safe.
    pub async fn read_chunks(&mut self, bufs: &mut [Bytes]) -> Result<Option<usize>, ReadError> {
        ReadChunks { stream: self, bufs }.await
    }

    /// Foundation of [`Self::read_chunks`]
    pub(super) fn poll_read_chunks(
        &mut self,
        cx: &mut Context,
        bufs: &mut [Bytes],
    ) -> Poll<Result<Option<usize>, ReadError>> {
        if bufs.is_empty() {
            return Poll::Ready(Ok(Some(0)));
        }

        self.poll_read_generic(cx, true, |chunks| {
            let mut read = 0;
            loop {
                if read >= bufs.len() {
                    // We know `read > 0` because `bufs` cannot be empty here
                    return ReadStatus::Readable(read);
                }

                match chunks.next(usize::MAX) {
                    Ok(Some(chunk)) => {
                        bufs[read] = chunk.bytes;
                        read += 1;
                    }
                    res => return (if read == 0 { None } else { Some(read) }, res.err()).into(),
                }
            }
        })
    }

    /// Convenience method to read all remaining data into a buffer
    ///
    /// Fails with [`ReadToEndError::TooLong`] on reading more than `size_limit` bytes, discarding
    /// all data read. Uses unordered reads to be more efficient than using `AsyncRead` would
    /// allow. `size_limit` should be set to limit worst-case memory use.
    ///
    /// If unordered reads have already been made, the resulting buffer may have gaps containing
    /// arbitrary data.
    ///
    /// This operation is *not* cancel-safe.
    ///
    /// [`ReadToEndError::TooLong`]: crate::ReadToEndError::TooLong
    pub async fn read_to_end(&mut self, size_limit: usize) -> Result<Vec<u8>, ReadToEndError> {
        ReadToEnd {
            stream: self,
            buffer: ReadToEndBuffer::new(size_limit),
        }
        .await
    }

    /// Stop accepting data
    ///
    /// Discards unread data and notifies the peer to stop transmitting. Once stopped, further
    /// attempts to operate on a stream will yield `ClosedStream` errors.
    pub fn stop(&mut self, error_code: VarInt) -> Result<(), ClosedStream> {
        let mut conn = self.conn.state.lock("RecvStream::stop");
        if self.is_0rtt && conn.check_0rtt().is_err() {
            return Ok(());
        }
        conn.inner.recv_stream(self.stream).stop(error_code)?;
        conn.wake();
        self.all_data_read = true;
        // Clean up shared state that might be left over from a cancalled read
        // operation, so `drop` doesn't have to
        conn.blocked_readers.remove(&self.stream);
        Ok(())
    }

    /// Check if this stream has been opened during 0-RTT.
    ///
    /// In which case any non-idempotent request should be considered dangerous at the application
    /// level. Because read data is subject to replay attacks.
    pub fn is_0rtt(&self) -> bool {
        self.is_0rtt
    }

    /// Get the identity of this stream
    pub fn id(&self) -> StreamId {
        self.stream
    }

    /// Completes when the stream has been reset by the peer or otherwise closed
    ///
    /// Yields `Some` with the reset error code when the stream is reset by the peer. Yields `None`
    /// when the stream was previously [`stop()`](Self::stop)ed, or when the stream was
    /// [`finish()`](crate::SendStream::finish)ed by the peer and all data has been received, after
    /// which it is no longer meaningful for the stream to be reset.
    ///
    /// This operation is cancel-safe.
    pub async fn received_reset(&mut self) -> Result<Option<VarInt>, ResetError> {
        poll_fn(|cx| {
            let mut conn = self.conn.state.lock("RecvStream::reset");
            if self.is_0rtt && conn.check_0rtt().is_err() {
                return Poll::Ready(Err(ResetError::ZeroRttRejected));
            }

            if let Some(code) = self.reset {
                return Poll::Ready(Ok(Some(code)));
            }

            match conn.inner.recv_stream(self.stream).received_reset() {
                Err(_) => Poll::Ready(Ok(None)),
                Ok(Some(error_code)) => {
                    // Stream state has just now been freed, so the connection may need to issue new
                    // stream ID flow control credit
                    conn.wake();
                    Poll::Ready(Ok(Some(error_code)))
                }
                Ok(None) => {
                    if let Some(e) = &conn.error {
                        return Poll::Ready(Err(e.clone().into()));
                    }
                    // Resets always notify readers, since a reset is an immediate read error. We
                    // could introduce a dedicated channel to reduce the risk of spurious wakeups,
                    // but that increased complexity is probably not justified, as an application
                    // that is expecting a reset is not likely to receive large amounts of data.
                    conn.blocked_readers.insert(self.stream, cx.waker().clone());
                    Poll::Pending
                }
            }
        })
        .await
    }

    /// Handle common logic related to reading out of a receive stream
    ///
    /// This takes an `FnMut` closure that takes care of the actual reading process, matching
    /// the detailed read semantics for the calling function with a particular return type.
    /// The closure can read from the passed `&mut Chunks` and has to return the status after
    /// reading: the amount of data read, and the status after the final read call.
    pub(super) fn poll_read_generic<T, U>(
        &mut self,
        cx: &mut Context,
        ordered: bool,
        mut read_fn: T,
    ) -> Poll<Result<Option<U>, ReadError>>
    where
        T: FnMut(&mut Chunks) -> ReadStatus<U>,
    {
        use proto::ReadError::*;
        if self.all_data_read {
            return Poll::Ready(Ok(None));
        }

        let mut conn = self.conn.state.lock("RecvStream::poll_read");
        if self.is_0rtt {
            conn.check_0rtt().map_err(|()| ReadError::ZeroRttRejected)?;
        }

        // If we stored an error during a previous call, return it now. This can happen if a
        // `read_fn` both wants to return data and also returns an error in its final stream status.
        let status = match self.reset {
            Some(code) => ReadStatus::Failed(None, Reset(code)),
            None => {
                let mut recv = conn.inner.recv_stream(self.stream);
                let mut chunks = recv.read(ordered)?;
                let status = read_fn(&mut chunks);
                if chunks.finalize().should_transmit() {
                    conn.wake();
                }
                status
            }
        };

        match status {
            ReadStatus::Readable(read) => Poll::Ready(Ok(Some(read))),
            ReadStatus::Finished(read) => {
                self.all_data_read = true;
                Poll::Ready(Ok(read))
            }
            ReadStatus::Failed(read, Blocked) => match read {
                Some(val) => Poll::Ready(Ok(Some(val))),
                None => {
                    if let Some(ref x) = conn.error {
                        return Poll::Ready(Err(ReadError::ConnectionLost(x.clone())));
                    }
                    conn.blocked_readers.insert(self.stream, cx.waker().clone());
                    Poll::Pending
                }
            },
            ReadStatus::Failed(read, Reset(error_code)) => match read {
                None => {
                    self.all_data_read = true;
                    self.reset = Some(error_code);
                    Poll::Ready(Err(ReadError::Reset(error_code)))
                }
                done => {
                    self.reset = Some(error_code);
                    Poll::Ready(Ok(done))
                }
            },
        }
    }
}
