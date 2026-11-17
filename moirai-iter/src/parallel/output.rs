//! Panic-safe direct initialization for ordered parallel map output.

use core::{
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    num::NonZeroUsize,
    ops::Range,
    ptr,
};

/// Derive one in-bounds output range without overflowing at the slice limit.
fn output_chunk_range(len: usize, chunk_size: usize, chunk_index: usize) -> Range<usize> {
    assert!(
        chunk_size > 0,
        "invariant: parallel map chunk size must be positive"
    );
    let start = chunk_index
        .checked_mul(chunk_size)
        .expect("invariant: parallel map chunk start is representable");
    let remaining = len
        .checked_sub(start)
        .expect("invariant: parallel map chunk starts in bounds");
    let chunk_len = remaining.min(chunk_size);
    let end = start
        .checked_add(chunk_len)
        .expect("invariant: in-bounds parallel map chunk end is representable");
    start..end
}

/// One final output allocation plus per-chunk completion ownership.
pub(crate) struct MapOutput<T> {
    values: Vec<MaybeUninit<T>>,
    completed: Vec<usize>,
    chunk_size: usize,
}

impl<T> MapOutput<T> {
    /// Allocate the final output and initialized completion slots.
    pub(crate) fn new(len: usize, chunk_size: NonZeroUsize) -> Self {
        let chunk_size = chunk_size.get();
        let mut values = Vec::with_capacity(len);
        // SAFETY: `MaybeUninit<T>` has no initialization validity requirement,
        // and the vector has capacity for exactly `len` logical slots.
        unsafe {
            values.set_len(len);
        }

        Self {
            values,
            completed: core::iter::repeat_n(0, len.div_ceil(chunk_size)).collect(),
            chunk_size,
        }
    }

    /// Return the number of disjoint output chunks.
    pub(crate) fn chunk_count(&self) -> usize {
        self.completed.len()
    }

    /// Return the final-storage pointer used by disjoint chunk writers.
    #[cfg(test)]
    fn values_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.values.as_mut_ptr()
    }

    /// Return the per-chunk completion pointer.
    #[cfg(test)]
    fn completed_ptr(&mut self) -> *mut usize {
        self.completed.as_mut_ptr()
    }

    /// Borrow the output and completion slots for indexed initialization.
    pub(crate) fn chunks(&mut self) -> MapOutputChunks<'_, T> {
        let len = self.values.len();
        MapOutputChunks {
            values: self.values.as_mut_ptr(),
            completed: self.completed.as_mut_ptr(),
            len,
            chunk_size: self.chunk_size,
            borrow: PhantomData,
        }
    }

    /// Convert the fully initialized storage without allocating or copying.
    ///
    /// # Panics
    ///
    /// Panics if any chunk has not completely initialized its slots.
    #[track_caller]
    pub(crate) fn into_vec(mut self) -> Vec<T> {
        let mut next_start = 0;
        for (chunk_index, &completed_end) in self.completed.iter().enumerate() {
            let range_start = chunk_index
                .checked_mul(self.chunk_size)
                .expect("invariant: parallel map chunk start is representable");
            assert_eq!(
                range_start, next_start,
                "invariant: parallel map chunks completed in source-slot order"
            );
            assert!(
                completed_end > range_start && completed_end <= self.values.len(),
                "invariant: every parallel map chunk completed in bounds"
            );
            next_start = completed_end;
        }
        assert_eq!(
            next_start,
            self.values.len(),
            "invariant: parallel map chunks initialized the complete output"
        );

        self.completed.clear();
        let mut values = ManuallyDrop::new(mem::take(&mut self.values));
        // SAFETY: the checked completion ranges are contiguous from zero to
        // `values.len()`, and a range is published only after its writer has
        // initialized every slot. `MaybeUninit<T>` has the same layout as `T`;
        // `ManuallyDrop` transfers the allocation to the returned vector.
        unsafe {
            Vec::from_raw_parts(
                values.as_mut_ptr().cast::<T>(),
                values.len(),
                values.capacity(),
            )
        }
    }
}

impl<T> Drop for MapOutput<T> {
    fn drop(&mut self) {
        for (chunk_index, &completed_end) in self.completed.iter().enumerate() {
            if completed_end == 0 {
                continue;
            }
            let range_start = chunk_index * self.chunk_size;
            debug_assert!(completed_end > range_start);
            debug_assert!(completed_end <= self.values.len());
            for index in range_start..completed_end {
                // SAFETY: a completion range is published only after all of
                // its slots are initialized. Ranges belong to distinct chunk
                // slots and therefore never overlap.
                unsafe {
                    ptr::drop_in_place(self.values.as_mut_ptr().add(index).cast::<T>());
                }
            }
        }
    }
}

/// A borrowed view used by jobs that own disjoint output chunk indices.
pub(crate) struct MapOutputChunks<'a, T> {
    values: *mut MaybeUninit<T>,
    completed: *mut usize,
    len: usize,
    chunk_size: usize,
    borrow: PhantomData<&'a mut MapOutput<T>>,
}

// SAFETY: the borrow keeps both allocations alive and writes are exclusive
// by chunk index. T: Send permits owned values to cross lanes.
unsafe impl<T: Send> Send for MapOutputChunks<'_, T> {}
// SAFETY: shared use exposes only layout and disjoint writes under the
// exclusive-index contract. No shared references to T are exposed.
unsafe impl<T: Send> Sync for MapOutputChunks<'_, T> {}

impl<T> MapOutputChunks<'_, T> {
    /// Return the logical element range for an output chunk.
    pub(crate) fn range(&self, index: usize) -> Option<Range<usize>> {
        (index < self.len.div_ceil(self.chunk_size))
            .then(|| output_chunk_range(self.len, self.chunk_size, index))
    }

    /// Initialize one chunk and transfer its values to the result owner.
    ///
    /// # Safety
    ///
    /// The caller must give this write exclusive access to its index,
    /// including against writes through other shared references.
    ///
    /// # Panics
    ///
    /// Panics for an invalid index, a completed chunk, or an iterator whose
    /// yield count differs from the chunk length. A panicking iterator leaves
    /// no values owned by the chunk.
    #[track_caller]
    pub(crate) unsafe fn write_chunk(&self, index: usize, values: impl IntoIterator<Item = T>) {
        let range = self
            .range(index)
            .expect("invariant: indexed work is within the output layout");
        // SAFETY: the validated index gives an in-bounds completion slot and
        // range. The caller guarantees exclusive access to that index.
        let (completed, mut writer) = unsafe {
            (
                &mut *self.completed.add(index),
                ChunkWriter::new(self.values, range),
            )
        };
        assert_eq!(*completed, 0, "invariant: the chunk is unpublished");

        for value in values {
            assert!(
                writer.initialized < writer.len,
                "invariant: the mapper yields at most one value per output slot"
            );
            // SAFETY: the writer retains an uninitialized slot and owns its range.
            unsafe { writer.push(value) };
        }

        *completed = writer.finish();
    }
}

/// Owns cleanup for one chunk until that chunk publishes completion.
struct ChunkWriter<T> {
    /// Pointer to the first slot owned by this writer.
    values: *mut MaybeUninit<T>,
    len: usize,
    completed_end: usize,
    initialized: usize,
    armed: bool,
}

impl<T> ChunkWriter<T> {
    /// Create a writer for one exclusive in-bounds output range.
    ///
    /// # Safety
    ///
    /// `values` must remain valid for the complete range until this writer is
    /// dropped. No other writer may access that range.
    unsafe fn new(values: *mut MaybeUninit<T>, range: Range<usize>) -> Self {
        let len = range
            .end
            .checked_sub(range.start)
            .expect("invariant: parallel map output range is ordered");
        // SAFETY: the caller guarantees that the complete range lies inside
        // the live output allocation.
        let values = unsafe { values.add(range.start) };
        Self {
            values,
            len,
            completed_end: range.end,
            initialized: 0,
            armed: true,
        }
    }

    /// Initialize the next logical slot in this writer's range.
    ///
    /// # Safety
    ///
    /// The writer must contain at least one uninitialized slot. Each call
    /// consumes exactly one such slot.
    #[inline]
    unsafe fn push(&mut self, value: T) {
        debug_assert!(self.initialized < self.len);
        // SAFETY: the caller guarantees an uninitialized slot remains, and
        // this writer has exclusive access to its local range.
        unsafe {
            self.values
                .add(self.initialized)
                .write(MaybeUninit::new(value));
        }
        self.initialized += 1;
    }

    /// Publish ownership of a completely initialized chunk range.
    fn finish(mut self) -> usize {
        assert_eq!(
            self.initialized, self.len,
            "invariant: parallel map chunk initialized every output slot"
        );
        self.armed = false;
        self.completed_end
    }
}

impl<T> Drop for ChunkWriter<T> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        for offset in 0..self.initialized {
            // SAFETY: `initialized` advances only after `push` writes the slot,
            // and this writer retains exclusive ownership until `finish`.
            unsafe {
                ptr::drop_in_place(self.values.add(offset).cast::<T>());
            }
        }
    }
}

#[cfg(test)]
mod tests;
