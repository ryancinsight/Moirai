//! One bounded physical writer and JSON decoder for independent trace owners.

use serde_json::Value;
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

pub(super) struct Buffer {
    pub(super) bytes: Vec<u8>,
    limit: usize,
}

pub(super) fn buffer(limit: usize) -> Arc<Mutex<Buffer>> {
    Arc::new(Mutex::new(Buffer {
        bytes: Vec::with_capacity(limit),
        limit,
    }))
}

pub(super) struct Capture(pub(super) Arc<Mutex<Buffer>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut output = self
            .0
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        if bytes.len() > output.limit - output.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "native diagnostic formatter exceeds the derived trace bound",
            ));
        }
        output.bytes.extend_from_slice(bytes);
        drop(output);
        io::stderr().write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        drop(
            self.0
                .lock()
                .map_err(|error| io::Error::other(error.to_string()))?,
        );
        io::stderr().flush()
    }
}

pub(super) fn observations(
    bytes: &[u8],
    target: &str,
    level: &str,
    event: &str,
    envelope: usize,
) -> Vec<Value> {
    bytes
        .split_inclusive(|byte| *byte == b'\n')
        .map(|line| {
            assert_eq!(
                line.last(),
                Some(&b'\n'),
                "complete native diagnostic record"
            );
            let mut value: Value = serde_json::from_slice(line).expect("actual formatter JSON");
            assert_eq!(value["target"], target);
            assert_eq!(value["level"], level);
            assert_eq!(value["fields"]["event"], event);
            let encoded = value["fields"]["observation"]
                .as_str()
                .expect("opaque encoded observation")
                .to_owned();
            assert_eq!(
                line.len(),
                serde_json::to_vec(&value)
                    .expect("compact physical envelope")
                    .len()
                    + 1
            );
            assert!(
                line.len() <= envelope + 2 * encoded.len(),
                "actual second JSON-string escaping bound"
            );
            value["fields"]["observation"] = Value::String(String::new());
            assert_eq!(
                serde_json::to_vec(&value)
                    .expect("actual empty wire envelope")
                    .len()
                    + 1,
                envelope
            );
            serde_json::from_str(&encoded).expect("actual native observation JSON")
        })
        .collect()
}
