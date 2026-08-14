//! Cache alignment utilities for performance optimization.
//!
//! This module is the single source of truth for the two cache granularities
//! the workspace reasons about. They are *different numbers* on the same
//! target and conflating them is a defect in both directions:
//!
//! * [`CACHE_LINE_SIZE`] — the coherence/transfer granularity of one line.
//!   Correct for prefetch strides and for sizing an iteration chunk so a block
//!   of elements lands in one line. 64 bytes on x86-64 and aarch64.
//! * [`DESTRUCTIVE_INTERFERENCE_SIZE`] — the separation two independently
//!   written atomics need to avoid false sharing. On x86-64 and modern aarch64
//!   the adjacent-line prefetcher pulls lines in pairs, so two objects 64 bytes
//!   apart still ping-pong a shared 128-byte sector; separation is 128 bytes.
//!   Correct for padding and alignment. Never for chunk sizing — doubling a
//!   chunk width silently doubles a kernel's working set.
//!
//! The per-target values follow the table in `crossbeam-utils`'
//! `CachePadded` (crossbeam-utils 0.8, `src/cache_padded.rs`), which documents
//! the vendor sources for each architecture.

/// Coherence and transfer granularity of a single cache line, in bytes.
///
/// Use for prefetch strides and for deriving how many elements of a type share
/// one line. For separating concurrently written data, use
/// [`DESTRUCTIVE_INTERFERENCE_SIZE`] instead — it is larger on the targets
/// whose prefetcher operates on line pairs.
pub const CACHE_LINE_SIZE: usize = if cfg!(any(
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "hexagon",
)) {
    32
} else if cfg!(target_arch = "m68k") {
    16
} else if cfg!(target_arch = "s390x") {
    256
} else if cfg!(target_arch = "powerpc64") {
    128
} else {
    // x86, x86-64, aarch64, riscv, loongarch, wasm and every target without a
    // documented deviation.
    64
};

/// Separation, in bytes, at which two concurrently written objects stop
/// interfering — the C++ `hardware_destructive_interference_size`.
///
/// Larger than [`CACHE_LINE_SIZE`] wherever the hardware prefetcher fetches
/// adjacent lines in pairs (x86-64, aarch64, powerpc64: 128 bytes), so padding
/// to one line is not enough to stop the ping-pong. This is the value
/// [`CacheAligned`] aligns to.
pub const DESTRUCTIVE_INTERFERENCE_SIZE: usize = if cfg!(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
)) {
    // Intel/AMD L2 spatial ("adjacent line") prefetcher and Apple M-series
    // both operate on 128-byte sectors.
    128
} else if cfg!(any(
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "hexagon",
)) {
    32
} else if cfg!(target_arch = "m68k") {
    16
} else if cfg!(target_arch = "s390x") {
    256
} else {
    64
};

// The `#[repr(align(..))]` attribute on `CacheAligned` takes a literal, so the
// table above is mirrored in `cfg_attr` form below. These assertions make the
// two representations impossible to drift apart: a target added to one table
// and not the other fails the build.
const _: () = assert!(core::mem::align_of::<CacheAligned<u8>>() == DESTRUCTIVE_INTERFERENCE_SIZE);
const _: () = assert!(CACHE_LINE_SIZE <= DESTRUCTIVE_INTERFERENCE_SIZE);
const _: () = assert!(CACHE_LINE_SIZE.is_power_of_two());
const _: () = assert!(DESTRUCTIVE_INTERFERENCE_SIZE.is_power_of_two());

/// Round a size up to the next cache-line boundary.
///
/// Uses the transfer granularity, not the false-sharing separation: this
/// answers "how many lines does this occupy", not "how far apart must these
/// live".
#[must_use]
pub const fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// A wrapper that pushes its value onto its own false-sharing sector.
///
/// Aligns the wrapped value to [`DESTRUCTIVE_INTERFERENCE_SIZE`] so an atomic
/// written by one core cannot invalidate a neighbouring atomic written by
/// another. Because a type's size is a multiple of its alignment, the wrapper
/// also *occupies* that many bytes — which is the point: the padding after the
/// value is what keeps the next field off this sector.
///
/// Structural traits are derived so the wrapper inherits them whenever the
/// inner `T` supports them.
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
    ),
    repr(align(128))
)]
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "sparc",
        target_arch = "hexagon",
    ),
    repr(align(32))
)]
#[cfg_attr(target_arch = "m68k", repr(align(16)))]
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "sparc",
        target_arch = "hexagon",
        target_arch = "m68k",
        target_arch = "s390x",
    )),
    repr(align(64))
)]
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheAligned<T>(pub T);

/// Zero-sized alignment marker.
///
/// Embedding one field of this type raises the containing struct's alignment
/// to [`DESTRUCTIVE_INTERFERENCE_SIZE`] without adding a byte, so a struct that
/// must start on its own sector single-sources the per-target table instead of
/// repeating a `#[repr(align(..))]` literal.
pub type CachePad = CacheAligned<()>;

const _: () = assert!(core::mem::size_of::<CachePad>() == 0);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Get a reference to the inner value.
    pub const fn get(&self) -> &T {
        &self.0
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> core::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for CacheAligned<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CacheAligned")
            .field("value", &self.0)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_to_cache_line_rounds_up_to_the_transfer_granularity() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(
            align_to_cache_line(CACHE_LINE_SIZE + 1),
            CACHE_LINE_SIZE * 2
        );
    }

    #[test]
    fn cache_aligned_wrapper_forwards_to_the_inner_value() {
        let aligned = CacheAligned::new(42);
        assert_eq!(*aligned, 42);
        assert_eq!(aligned.get(), &42);
    }

    #[test]
    fn cache_aligned_separates_neighbours_by_the_interference_size() {
        // Two wrapped atomics written by different cores must not share a
        // sector: the byte distance between consecutive elements is the
        // separation the wrapper promises.
        let pair = [CacheAligned::new(0_u8), CacheAligned::new(0_u8)];
        let first = std::ptr::addr_of!(pair[0].0) as usize;
        let second = std::ptr::addr_of!(pair[1].0) as usize;
        assert_eq!(second - first, DESTRUCTIVE_INTERFERENCE_SIZE);
    }

    #[test]
    fn cache_pad_is_free_but_raises_alignment() {
        struct Host {
            _pad: CachePad,
            value: u8,
        }

        assert_eq!(core::mem::size_of::<CachePad>(), 0);
        assert_eq!(core::mem::align_of::<Host>(), DESTRUCTIVE_INTERFERENCE_SIZE);
        let host = Host {
            _pad: CacheAligned::new(()),
            value: 7,
        };
        assert_eq!(host.value, 7);
    }

    /// The relation between the two constants is pinned by the `const _`
    /// assertions at module scope; this pins the *values* on the targets where
    /// a regression to a single 64-byte constant is the likely mistake.
    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn line_and_interference_sizes_differ_on_this_target() {
        assert_eq!(CACHE_LINE_SIZE, 64);
        assert_eq!(DESTRUCTIVE_INTERFERENCE_SIZE, 128);
        assert_eq!(
            core::mem::align_of::<CacheAligned<u8>>(),
            DESTRUCTIVE_INTERFERENCE_SIZE
        );
    }
}
