//! Platform abstraction layer for cross-platform compatibility.
//!
//! `core` and `alloc` vocabulary is available on every supported target.
//! Host-provided collections, locks, channels, threads, and monotonic time are
//! exported only with the `std` feature.

pub use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
pub use core::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering, compiler_fence, fence,
};
pub use core::time::Duration;
#[cfg(feature = "std")]
pub use std::{collections::HashMap, time::Instant};

pub use core::cell::{Cell, RefCell, UnsafeCell};
pub use core::fmt::{self, Debug, Display};
pub use core::marker::{PhantomData, Send, Sync};
pub use core::mem::{self, MaybeUninit, align_of, forget, replace, size_of, swap};
pub use core::ops::{Deref, DerefMut, Drop, Fn, FnMut, FnOnce};
pub use core::ptr::{self, NonNull, null, null_mut};

// Mutex abstraction
#[cfg(feature = "std")]
pub use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Channel abstraction for cross-platform message passing.
///
/// This module re-exports the standard library's multi-producer,
/// single-consumer channel vocabulary.
#[cfg(feature = "std")]
pub mod channel {
    pub use std::sync::mpsc::{Receiver, RecvError, Sender, TryRecvError, channel};
}

/// Thread abstraction for cross-platform threading support.
///
/// This module re-exports standard thread handles and scheduling operations.
#[cfg(feature = "std")]
pub mod thread {
    pub use std::thread::{JoinHandle, Thread, ThreadId, sleep, spawn, yield_now};
}
