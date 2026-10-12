//! One owned byte-stream reader for selected files and complete HTTP bodies.
use crate::http_policy::count_chunk;
use std::io;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

pub(super) trait Sink {
    const STOP_WHEN_FULL: bool;
    fn bound(&self) -> usize;
    fn copy_chunk(&mut self, start: usize, end: usize, chunk: &js_sys::Uint8Array);
}

pub(super) struct SliceSink<'a>(pub(super) &'a mut [u8]);
impl Sink for SliceSink<'_> {
    const STOP_WHEN_FULL: bool = true;
    fn bound(&self) -> usize {
        self.0.len()
    }
    fn copy_chunk(&mut self, start: usize, end: usize, chunk: &js_sys::Uint8Array) {
        chunk.copy_to(&mut self.0[start..end]);
    }
}

pub(super) struct VecSink {
    pub(super) bytes: Vec<u8>,
    pub(super) bound: usize,
}
impl Sink for VecSink {
    const STOP_WHEN_FULL: bool = false;
    fn bound(&self) -> usize {
        self.bound
    }
    fn copy_chunk(&mut self, start: usize, end: usize, chunk: &js_sys::Uint8Array) {
        self.bytes.reserve_exact(end - self.bytes.len());
        self.bytes.resize(end, 0);
        chunk.copy_to(&mut self.bytes[start..end]);
    }
}

pub(super) async fn read<S: Sink>(
    stream: web_sys::ReadableStream,
    sink: &mut S,
) -> io::Result<usize> {
    let reader = stream
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| io::Error::other("Browser rejected the byte-stream reader"))?;
    let mut reader = Reader {
        reader,
        finished: false,
    };
    let mut copied = 0;
    loop {
        if S::STOP_WHEN_FULL && copied == sink.bound() {
            break;
        }
        let result = JsFuture::from(reader.reader.read())
            .await
            .map_err(|_| io::Error::other("Browser rejected the byte-stream read"))?;
        // ReadableStreamReadResult is a Web IDL dictionary, with no runtime
        // constructor for instanceof. Validate object shape, then validate
        // done and value below instead of rejecting every native dictionary.
        if !result.is_object() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Browser returned a non-object stream result",
            ));
        }
        let result = result.unchecked_into::<web_sys::ReadableStreamReadResult>();
        let done = result.get_done().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Browser byte stream omitted its completion flag",
            )
        })?;
        let value = result.get_value();
        if !value.is_undefined() && !value.is_null() {
            let chunk = value.dyn_into::<js_sys::Uint8Array>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Browser byte stream returned a non-byte chunk",
                )
            })?;
            let length = usize::try_from(chunk.length()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Browser chunk length cannot be represented",
                )
            })?;
            if length == 0 && !done {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Browser byte stream returned an empty incomplete chunk",
                ));
            }
            let end = count_chunk(copied, length, sink.bound())?;
            sink.copy_chunk(copied, end, &chunk);
            copied = end;
        } else if !done {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Browser byte stream omitted an incomplete chunk",
            ));
        }
        if done {
            reader.finished = true;
            break;
        }
    }
    Ok(copied)
}

struct Reader {
    reader: web_sys::ReadableStreamDefaultReader,
    finished: bool,
}
impl Drop for Reader {
    fn drop(&mut self) {
        if !self.finished {
            // Observe asynchronous cancellation rejection without retaining a
            // callback or producing an unhandled Promise rejection.
            let cancelled = JsFuture::from(self.reader.cancel());
            wasm_bindgen_futures::spawn_local(async move {
                let _ = cancelled.await;
            });
        }
        self.reader.release_lock();
    }
}
