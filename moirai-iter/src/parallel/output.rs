//! Panic-safe direct initialization for ordered parallel map output.

use std::{
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::Range,
    ptr,
};

/// Derive one in-bounds output range without overflowing at the slice limit.
pub(crate) fn output_chunk_range(
    len: usize,
    chunk_size: usize,
    chunk_index: usize,
) -> Range<usize> {
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
    pub(crate) fn new(len: usize, chunk_size: usize) -> Self {
        assert!(
            chunk_size > 0,
            "invariant: parallel map chunk size must be positive"
        );
        let mut values = Vec::with_capacity(len);
        // SAFETY: `MaybeUninit<T>` has no initialization validity requirement,
        // and the vector has capacity for exactly `len` logical slots.
        unsafe {
            values.set_len(len);
        }

        Self {
            values,
            completed: vec![0; len.div_ceil(chunk_size)],
            chunk_size,
        }
    }

    /// Return the number of disjoint output chunks.
    pub(crate) fn chunk_count(&self) -> usize {
        self.completed.len()
    }

    /// Return the final-storage pointer used by disjoint chunk writers.
    pub(crate) fn values_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.values.as_mut_ptr()
    }

    /// Return the per-chunk completion pointer.
    pub(crate) fn completed_ptr(&mut self) -> *mut usize {
        self.completed.as_mut_ptr()
    }

    /// Convert the fully initialized storage without allocating or copying.
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

/// Owns cleanup for one chunk until that chunk publishes completion.
pub(crate) struct ChunkWriter<T> {
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
    pub(crate) unsafe fn new(values: *mut MaybeUninit<T>, range: Range<usize>) -> Self {
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
    pub(crate) unsafe fn push(&mut self, value: T) {
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
    pub(crate) fn finish(mut self) -> usize {
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
mod tests {
    use super::{output_chunk_range, ChunkWriter, MapOutput};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Tracked<'a>(&'a AtomicUsize);

    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn completed_chunks_transfer_the_original_allocation() {
        let mut output = MapOutput::new(5, 3);
        let values = output.values_ptr();
        let completed = output.completed_ptr();

        // SAFETY: the two writers receive disjoint ranges within the five-slot
        // allocation, and both writers finish before the allocation is moved.
        let first_end = unsafe {
            let mut writer = ChunkWriter::new(values, 0..3);
            writer.push(10);
            writer.push(11);
            writer.push(12);
            writer.finish()
        };
        // SAFETY: this range is disjoint from the first writer and remains
        // within the same live five-slot allocation.
        let second_end = unsafe {
            let mut writer = ChunkWriter::new(values, 3..5);
            writer.push(13);
            writer.push(14);
            writer.finish()
        };
        // SAFETY: both indices belong to the live two-slot completion array;
        // each range is published exactly once after its writer finishes.
        unsafe {
            completed.write(first_end);
            completed.add(1).write(second_end);
        }

        let result = output.into_vec();
        assert_eq!(result.as_ptr(), values.cast::<i32>());
        assert_eq!(result, [10, 11, 12, 13, 14]);
    }

    #[test]
    fn unfinished_writer_drops_only_its_initialized_prefix() {
        let drops = AtomicUsize::new(0);
        let mut output = MapOutput::new(4, 4);

        // SAFETY: the writer owns the complete live four-slot allocation and
        // is dropped before the allocation itself.
        let mut writer = unsafe { ChunkWriter::new(output.values_ptr(), 0..4) };
        // SAFETY: the writer owns four uninitialized slots and receives only
        // two values before it is dropped.
        unsafe {
            writer.push(Tracked(&drops));
            writer.push(Tracked(&drops));
        }
        drop(writer);

        assert_eq!(drops.load(Ordering::Relaxed), 2);
        drop(output);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn mapper_panic_drops_current_prefix_and_completed_peers_once() {
        let drops = AtomicUsize::new(0);
        let mut output = MapOutput::new(6, 3);
        let values = output.values_ptr();
        let completed = output.completed_ptr();

        // SAFETY: the first writer exclusively owns the first three live slots
        // and initializes the complete range before publishing it.
        let completed_end = unsafe {
            let mut writer = ChunkWriter::new(values, 0..3);
            writer.push(Tracked(&drops));
            writer.push(Tracked(&drops));
            writer.push(Tracked(&drops));
            writer.finish()
        };
        // SAFETY: index zero is a live completion slot and the first range is
        // fully initialized.
        unsafe {
            completed.write(completed_end);
        }

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the second writer exclusively owns the final three live
            // slots and is dropped while unwinding before the output owner.
            let mut writer = unsafe { ChunkWriter::new(values, 3..6) };
            // SAFETY: all three slots are initially uninitialized; only the
            // first is consumed before the simulated mapper panic.
            unsafe {
                writer.push(Tracked(&drops));
            }
            panic!("simulated mapper panic");
        }));

        let Err(payload) = panic else {
            panic!("invariant: the simulated mapper panic must unwind through the writer");
        };
        assert_eq!(
            crate::test_support::panic_message(payload.as_ref()),
            "simulated mapper panic"
        );
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(output);
        assert_eq!(drops.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn zero_sized_outputs_retain_their_logical_length() {
        let mut output = MapOutput::new(3, 3);
        let values = output.values_ptr();
        let completed = output.completed_ptr();

        // SAFETY: the writer exclusively owns the full logical range of the
        // live zero-sized allocation and initializes every logical slot.
        let completed_end = unsafe {
            let mut writer = ChunkWriter::new(values, 0..3);
            writer.push(());
            writer.push(());
            writer.push(());
            writer.finish()
        };
        // SAFETY: index zero is the sole live completion slot.
        unsafe {
            completed.write(completed_end);
        }

        assert_eq!(output.into_vec(), [(), (), ()]);
    }

    #[test]
    fn output_chunk_range_reaches_the_usize_limit_without_overflow() {
        let chunk_size = 1_024;
        let last_chunk = usize::MAX.div_ceil(chunk_size) - 1;

        assert_eq!(output_chunk_range(usize::MAX, chunk_size, 0), 0..1_024);
        assert_eq!(
            output_chunk_range(usize::MAX, chunk_size, last_chunk),
            (usize::MAX - 1_023)..usize::MAX
        );
        assert_eq!(output_chunk_range(usize::MAX, usize::MAX, 0), 0..usize::MAX);
    }
}
