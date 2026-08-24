//! Fuzz target for the HTTP/1.1 response codec.
//!
//! Drives `read_response` with arbitrary byte streams under a fixed
//! slowloris budget, exercising header parsing, Content-Length and chunked
//! body framing. A panic, a hang beyond libFuzzer's timeout, or an
//! allocation blowup inside the parser is a defect: malformed input must
//! surface as `io::ErrorKind::InvalidData`, never as UB or a crash.

#![no_main]

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use moirai_async::io::AsyncRead;
use moirai_http::codec::read_response;

struct FuzzReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl AsyncRead for FuzzReader<'_> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        if self.pos >= self.data.len() {
            return Poll::Ready(Ok(0));
        }
        let (_, rest) = self.data.split_at(self.pos);
        let n = rest.len().min(buf.len());
        let (chunk, _) = rest.split_at(n);
        buf.get_mut(..n)
            .expect("n <= buf.len() by the min above")
            .copy_from_slice(chunk);
        self.pos += n;
        Poll::Ready(Ok(n))
    }
}

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let budget = 1 << 20; // 1 MiB cap keeps any single input bounded
    let mut reader = FuzzReader { data, pos: 0 };
    // A parser that panics or aborts on hostile bytes fails the run; every
    // rejection path must be a typed io::Error instead.
    let _ = moirai_runtime::block_on(read_response(&mut reader, false, budget));
});
