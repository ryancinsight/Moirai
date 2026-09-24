//! Stream operations as jobs for the file-system pool, and what they return.

use std::io::{self, SeekFrom};

use moirai_pal::fs::File as Handle;

use crate::blocking::Abandoned;

/// One cursor-affecting operation, owning every buffer it touches so it can
/// run on a pool worker after its caller has gone.
pub(super) enum Request {
    /// Read up to `len` bytes at the cursor.
    Read { len: usize },
    /// Read from the cursor to the end, after `prefix`: bytes already read
    /// but not yet delivered.
    ReadToEnd { prefix: Vec<u8> },
    /// Move the cursor back `rewind` bytes, then write `data` once.
    Write { data: Vec<u8>, rewind: u64 },
    /// Move the cursor back `rewind` bytes, then write all of `data`.
    WriteAll { data: Vec<u8>, rewind: u64 },
    /// Move the cursor.
    Seek(SeekFrom),
    /// Synchronize data and metadata.
    SyncAll,
    /// Synchronize data.
    SyncData,
}

/// The result of a [`Request`].
pub(super) enum Outcome {
    /// Bytes read, empty at end of file.
    Read(io::Result<Vec<u8>>),
    /// Bytes written by a single write.
    Wrote(io::Result<usize>),
    /// The new cursor position.
    Sought(io::Result<u64>),
    /// A write-all or sync finished.
    Done(io::Result<()>),
}

impl Outcome {
    /// The error of a failed outcome whose caller has gone.
    pub(super) fn into_error(self) -> Option<io::Error> {
        match self {
            Self::Read(result) => result.err(),
            Self::Wrote(result) => result.err(),
            Self::Sought(result) => result.err(),
            Self::Done(result) => result.err(),
        }
    }
}

impl Request {
    /// A write's bytes must reach the file once submitted, even if every
    /// caller has gone. Any other request is only worth its result.
    pub(super) fn abandoned(&self) -> Abandoned {
        match self {
            Self::Write { .. } | Self::WriteAll { .. } => Abandoned::Run,
            Self::Read { .. }
            | Self::ReadToEnd { .. }
            | Self::Seek(_)
            | Self::SyncAll
            | Self::SyncData => Abandoned::Skip,
        }
    }

    pub(super) fn run(self, handle: &Handle) -> Outcome {
        match self {
            Self::Read { len } => Outcome::Read(read_up_to(handle, len)),
            Self::ReadToEnd { mut prefix } => {
                Outcome::Read(handle.read_to_end(&mut prefix).map(|_| prefix))
            }
            Self::Write { data, rewind } => {
                Outcome::Wrote(rewind_by(handle, rewind).and_then(|()| handle.write(&data)))
            }
            Self::WriteAll { data, rewind } => {
                Outcome::Done(rewind_by(handle, rewind).and_then(|()| handle.write_all(&data)))
            }
            Self::Seek(pos) => Outcome::Sought(handle.seek(pos)),
            Self::SyncAll => Outcome::Done(handle.sync_all()),
            Self::SyncData => Outcome::Done(handle.sync_data()),
        }
    }
}

/// Read at most `len` bytes at the cursor into a buffer of exactly the bytes
/// read.
fn read_up_to(handle: &Handle, len: usize) -> io::Result<Vec<u8>> {
    let mut data = vec![0; len];
    let read = handle.read(&mut data)?;
    data.truncate(read);
    Ok(data)
}

/// Move the cursor back over `rewind` bytes read ahead but never delivered.
fn rewind_by(handle: &Handle, rewind: u64) -> io::Result<()> {
    if rewind == 0 {
        return Ok(());
    }
    let offset = i64::try_from(rewind).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{rewind} undelivered bytes exceed a seek offset"),
        )
    })?;
    handle.seek(SeekFrom::Current(-offset)).map(drop)
}
