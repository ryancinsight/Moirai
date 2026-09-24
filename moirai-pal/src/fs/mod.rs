//! Platform file-system primitives.
//!
//! Every operation here is a blocking syscall. The async runtime runs them on a
//! bounded blocking pool (`moirai_async::fs`), never inside `poll`.

mod confined;
mod file;

pub use confined::open_file_within_root;
pub use file::{File, FileOpenOptions};

#[cfg(test)]
mod tests;
