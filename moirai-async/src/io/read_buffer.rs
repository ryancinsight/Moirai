//! Initialized-prefix ownership for native asynchronous reads.
//!
//! The invariant is `filled <= initialized <= capacity`. Append initializes
//! before advancing either prefix. Clearing only shrinks the filled prefix;
//! advancing cannot expose bytes beyond the initialized prefix. Borrowed
//! storage prevents mutation or deallocation by the caller until this view
//! is dropped. No allocator, runtime, or external I/O trait is required.

use std::mem::MaybeUninit;

/// A borrowed read destination tracking initialized and filled prefixes.
///
/// [`new`](Self::new) borrows initialized bytes; [`uninit`](Self::uninit)
/// borrows potentially uninitialized storage. [`put_slice`](Self::put_slice)
/// initializes exactly the appended bytes. Only the initialized prefix is
/// ever exposed as a byte slice, including after a partial read or error.
pub struct ReadBuffer<'a> {
    storage: &'a mut [MaybeUninit<u8>],
    filled: usize,
    initialized: usize,
}

impl<'a> ReadBuffer<'a> {
    /// Borrow initialized storage with an empty filled prefix.
    #[must_use]
    pub const fn new(storage: &'a mut [u8]) -> Self {
        let initialized = storage.len();
        // SAFETY: MaybeUninit<u8> has the same size/alignment as u8 and may
        // hold every u8 bit pattern. The exclusive borrowed slice has exactly
        // the original allocation, length and lifetime; all bytes are known
        // initialized. This conversion adds no allocation or representation tag.
        let storage = unsafe {
            std::slice::from_raw_parts_mut(
                storage.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                initialized,
            )
        };
        Self {
            storage,
            filled: 0,
            initialized,
        }
    }

    /// Borrow potentially uninitialized storage with both prefixes empty.
    #[must_use]
    pub const fn uninit(storage: &'a mut [MaybeUninit<u8>]) -> Self {
        Self {
            storage,
            filled: 0,
            initialized: 0,
        }
    }

    /// Total length of the borrowed destination.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Bytes available after the filled prefix, initialized or otherwise.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.capacity() - self.filled
    }

    /// The bytes already filled by the reader.
    #[must_use]
    pub fn filled(&self) -> &[u8] {
        &self.initialized()[..self.filled]
    }

    /// All bytes known to be initialized, including any unfilled prefix.
    #[must_use]
    pub fn initialized(&self) -> &[u8] {
        // SAFETY: new starts with an initialized slice; uninit starts with an
        // empty prefix. Only put_slice and initialize_unfilled grow this
        // prefix, after writing every exposed byte. The borrowed view remains
        // within the backing allocation and cannot outlive &self.
        unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), self.initialized) }
    }

    /// Initialize unknown bytes with zero and borrow the entire unfilled tail.
    ///
    /// Existing initialized bytes are preserved. Mutating this slice does not
    /// advance the filled prefix; call [`advance`](Self::advance) after writing.
    pub fn initialize_unfilled(&mut self) -> &mut [u8] {
        for byte in &mut self.storage[self.initialized..] {
            byte.write(0);
        }
        self.initialized = self.capacity();
        // SAFETY: the original prefix and every byte in the remaining tail
        // are initialized above. The exclusive view stays within the same
        // allocation and is borrowed for no longer than &mut self.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.storage.as_mut_ptr().cast::<u8>().add(self.filled),
                self.remaining(),
            )
        }
    }

    /// Append bytes, initializing only their destination and retaining prior data.
    ///
    /// # Panics
    ///
    /// Panics before modifying the buffer when `bytes.len() > remaining()`.
    pub fn put_slice(&mut self, bytes: &[u8]) {
        assert!(
            bytes.len() <= self.remaining(),
            "read buffer capacity exceeded"
        );
        let end = self.filled + bytes.len();
        for (destination, source) in self.storage[self.filled..end].iter_mut().zip(bytes) {
            destination.write(*source);
        }
        self.filled = end;
        self.initialized = self.initialized.max(end);
    }

    /// Mark `count` additional initialized bytes as filled.
    ///
    /// # Panics
    ///
    /// Panics without changing either prefix if the sum overflows or would
    /// expose an uninitialized byte. Initialize before advancing.
    pub fn advance(&mut self, count: usize) {
        let end = self
            .filled
            .checked_add(count)
            .expect("read buffer length overflow");
        assert!(
            end <= self.initialized,
            "cannot advance into uninitialized bytes"
        );
        self.filled = end;
    }

    /// Empty the filled prefix while retaining initialization for future reuse.
    pub const fn clear(&mut self) {
        self.filled = 0;
    }
}

#[cfg(test)]
#[path = "read_buffer_tests.rs"]
mod tests;
