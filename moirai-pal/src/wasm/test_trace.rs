//! Full span-tree evidence through the actual browser console.

use std::io::{self, Write};

#[derive(Default)]
struct BrowserWriter(Vec<u8>);

impl Write for BrowserWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.0.is_empty() {
            let text = std::str::from_utf8(&self.0)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(text));
            self.0.clear();
        }
        Ok(())
    }
}

impl Drop for BrowserWriter {
    fn drop(&mut self) {
        self.flush()
            .expect("Rust trace formatting emits valid UTF8");
    }
}

pub(crate) fn install() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            // std::time is absent on wasm32-unknown-unknown. Full lifecycle
            // spans retain their hierarchy without unsupported clock calls.
            .without_time()
            .with_writer(BrowserWriter::default)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}
