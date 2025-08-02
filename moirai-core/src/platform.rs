//! Platform abstraction layer for cross-platform compatibility.
//!
//! This module provides a unified interface for platform-specific functionality,
//! following the Dependency Inversion Principle (SOLID) and making the code more
//! composable (CUPID).

#![allow(unused_imports)]

// Re-export common types based on platform
#[cfg(feature = "std")]
pub use std::{
    boxed::Box,
    vec::Vec,
    string::String,
    sync::Arc,
    collections::HashMap,
    time::{Duration, Instant},
};

#[cfg(not(feature = "std"))]
pub use alloc::{
    boxed::Box,
    vec::Vec,
    string::String,
    sync::Arc,
    collections::HashMap,
};

#[cfg(not(feature = "std"))]
pub use core::time::Duration;

// Platform-specific atomic operations
#[cfg(feature = "std")]
pub use std::sync::atomic::{
    AtomicBool, AtomicUsize, AtomicU64, AtomicPtr,
    Ordering, fence, compiler_fence,
};

#[cfg(not(feature = "std"))]
pub use core::sync::atomic::{
    AtomicBool, AtomicUsize, AtomicU64, AtomicPtr,
    Ordering, fence, compiler_fence,
};

// Platform-specific cell types
#[cfg(feature = "std")]
pub use std::cell::{UnsafeCell, RefCell, Cell};

#[cfg(not(feature = "std"))]
pub use core::cell::{UnsafeCell, RefCell, Cell};

// Platform-specific memory operations
#[cfg(feature = "std")]
pub use std::mem::{self, MaybeUninit, size_of, align_of, forget, replace, swap};

#[cfg(not(feature = "std"))]
pub use core::mem::{self, MaybeUninit, size_of, align_of, forget, replace, swap};

// Platform-specific pointer operations
#[cfg(feature = "std")]
pub use std::ptr::{self, null, null_mut, NonNull};

#[cfg(not(feature = "std"))]
pub use core::ptr::{self, null, null_mut, NonNull};

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

// Thread-local storage abstraction

/// Creates a thread-local static variable with platform-specific implementation.
/// 
/// This macro provides a unified interface for thread-local storage across
/// different platforms (std and no_std environments).
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
#[macro_export]
macro_rules! thread_local_static {
    ($(#[$attr:meta])* static $name:ident: $t:ty = $init:expr) => {
        thread_local! {
            $(#[$attr])*
            static $name: $t = $init
        }
    };
}

#[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
#[macro_export]
macro_rules! thread_local_static {
    ($(#[$attr:meta])* static $name:ident: $t:ty = $init:expr) => {
        // For no-std or WASM, use a static variable (no thread-local support)
        $(#[$attr])*
        static $name: $t = $init;
    };
}

// Time abstraction for no-std environments
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(u64);

#[cfg(not(feature = "std"))]
impl Instant {
    /// Create a new instant representing "now"
    pub fn now() -> Self {
        // In no-std, we can't get real time, so use a counter
        // This is a simplified implementation
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
    pub use std::sync::mpsc::{channel, Sender, Receiver, RecvError, TryRecvError};
    
    #[cfg(not(feature = "std"))]
    pub use alloc::sync::mpsc::{channel, Sender, Receiver, RecvError, TryRecvError};
}

/// Thread abstraction for cross-platform threading support.
/// 
/// This module provides a unified interface for thread operations that works
/// across different platforms and feature configurations.
pub mod thread {
    #[cfg(feature = "std")]
    pub use std::thread::{spawn, sleep, yield_now, JoinHandle, Thread, ThreadId};
    
    #[cfg(not(feature = "std"))]
    compile_error!("Thread support requires std feature");
}