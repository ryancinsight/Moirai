use moirai::Moirai;
use moirai_core::TaskBuilder;
use moirai_utils::simd;
use rayon::prelude::*;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const READY_COUNT: usize = 257;
const MAP_REDUCE_COUNT: usize = 4_096;
const WORKER_THREADS: usize = 4;
const CPU_WORK: usize = 8;

/// Run-time path resolution, retained for the one audit that asserts a file's
/// *absence*: `include_str!` cannot express "this path must not exist", since
/// naming a missing file is a compile error rather than a readable fact. Every
/// audit that reads content goes through [`EMBEDDED_SOURCES`] instead.
///
/// `env!("CARGO_MANIFEST_DIR")` resolves when this binary is compiled, so this
/// single check still reports on whichever worktree last built the binary.
fn benchmark_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}
