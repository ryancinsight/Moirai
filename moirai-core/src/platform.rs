//! Platform abstraction layer for cross-platform compatibility.
//!
//! This module provides a unified interface for platform-specific functionality,
//! following the Dependency Inversion Principle (SOLID) and making the code more
//! composable (CUPID).

// Re-export common types based on platform
#[cfg(feature = "std")]
pub use std::{
    boxed::Box,
    collections::HashMap,
    string::String,
    sync::Arc,
    time::{Duration, Instant},
    vec::Vec,
};

#[cfg(not(feature = "std"))]
pub use alloc::{boxed::Box, collections::HashMap, string::String, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
pub use core::time::Duration;

// Platform-specific atomic operations
#[cfg(feature = "std")]
pub use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering, compiler_fence, fence,
};

#[cfg(not(feature = "std"))]
pub use core::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering, compiler_fence, fence,
};

// Platform-specific cell types
#[cfg(feature = "std")]
pub use std::cell::{Cell, RefCell, UnsafeCell};

#[cfg(not(feature = "std"))]
pub use core::cell::{Cell, RefCell, UnsafeCell};

// Platform-specific memory operations
#[cfg(feature = "std")]
pub use std::mem::{self, MaybeUninit, align_of, forget, replace, size_of, swap};

#[cfg(not(feature = "std"))]
pub use core::mem::{self, MaybeUninit, align_of, forget, replace, size_of, swap};

// Platform-specific pointer operations
#[cfg(feature = "std")]
pub use std::ptr::{self, NonNull, null, null_mut};

#[cfg(not(feature = "std"))]
pub use core::ptr::{self, NonNull, null, null_mut};

// Platform-specific marker types
#[cfg(feature = "std")]
pub use std::marker::{PhantomData, Send, Sync};

#[cfg(not(feature = "std"))]
pub use core::marker::{PhantomData, Send, Sync};

// Platform-specific ops
#[cfg(feature = "std")]
pub use std::ops::{Deref, DerefMut, Drop, Fn, FnMut, FnOnce};

#[cfg(not(feature = "std"))]
pub use core::ops::{Deref, DerefMut, Drop, Fn, FnMut, FnOnce};

// Platform-specific formatting
#[cfg(feature = "std")]
pub use std::fmt::{self, Debug, Display};

#[cfg(not(feature = "std"))]
pub use core::fmt::{self, Debug, Display};

// Time abstraction for no-std environments
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(u64);

#[cfg(not(feature = "std"))]
impl Instant {
    /// Create a new instant representing "now"
    pub fn now() -> Self {
        // In no-std, we can't get real time, so use a counter
        // Monotonic counter for simulating Instant::now() in no-std environments
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Instant(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get duration since another instant
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        let nanos = self.0.saturating_sub(earlier.0);
        Duration::from_nanos(nanos)
    }

    /// Get elapsed time since this instant
    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(*self)
    }
}

// Mutex abstraction
#[cfg(feature = "std")]
pub use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(not(feature = "std"))]
pub use spin::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Channel abstraction for cross-platform message passing.
///
/// This module provides a unified interface for channels that works
/// across different platforms and feature configurations.
pub mod channel {
    #[cfg(feature = "std")]
    pub use std::sync::mpsc::{Receiver, RecvError, Sender, TryRecvError, channel};

    #[cfg(not(feature = "std"))]
    pub use alloc::sync::mpsc::{Receiver, RecvError, Sender, TryRecvError, channel};
}

/// Thread abstraction for cross-platform threading support.
///
/// This module provides a unified interface for thread operations that works
/// across different platforms and feature configurations.
pub mod thread {
    #[cfg(feature = "std")]
    pub use std::thread::{JoinHandle, Thread, ThreadId, sleep, spawn, yield_now};

    #[cfg(not(feature = "std"))]
    compile_error!("Thread support requires std feature");
}
