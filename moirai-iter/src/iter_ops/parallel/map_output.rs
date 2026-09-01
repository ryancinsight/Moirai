//! Panic-safe direct initialization for ordered parallel map output.

use std::{
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::Range,
    ptr,
};

/// One final output allocation plus per-chunk completion ownership.
pub(super) struct MapOutput<T> {
    values: Vec<MaybeUninit<T>>,
    completed: Vec<Option<Range<usize>>>,
}

impl<T> MapOutput<T> {
    /// Allocate the final output and initialized completion slots.
    pub(super) fn new(len: usize, chunk_count: usize) -> Self {
        let mut values = Vec::with_capacity(len);
        // SAFETY: `MaybeUninit<T>` has no initialization validity requirement,
        // and the vector has capacity for exactly `len` logical slots.
        unsafe {
            values.set_len(len);
        }

        Self {
            values,
            completed: (0..chunk_count).map(|_| None).collect(),
        }
    }

    /// Return the final-storage pointer used by disjoint chunk writers.
    pub(super) fn values_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.values.as_mut_ptr()
    }

    /// Return the per-chunk completion pointer.
    pub(super) fn completed_ptr(&mut self) -> *mut Option<Range<usize>> {
        self.completed.as_mut_ptr()
    }

    /// Convert the fully initialized storage without allocating or copying.
    pub(super) fn into_vec(mut self) -> Vec<T> {
        let mut next_start = 0;
        for completed in &self.completed {
            let range = completed
                .as_ref()
                .expect("invariant: every parallel map chunk completed");
            assert_eq!(
                range.start, next_start,
                "invariant: parallel map chunks completed in source-slot order"
            );
            next_start = range.end;
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
        for range in self.completed.iter().flatten() {
            for index in range.clone() {
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
pub(super) struct ChunkWriter<T> {
    values: *mut MaybeUninit<T>,
    range: Range<usize>,
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
    pub(super) unsafe fn new(values: *mut MaybeUninit<T>, range: Range<usize>) -> Self {
        Self {
            values,
            range,
            initialized: 0,
            armed: true,
        }
    }

    /// Initialize the next logical slot in this writer's range.
    pub(super) fn push(&mut self, value: T) {
        let index = self
            .range
            .start
            .checked_add(self.initialized)
            .expect("invariant: parallel map output index is representable");
        assert!(
            index < self.range.end,
            "invariant: parallel map chunk cannot overrun its output range"
        );
        let next_initialized = self
            .initialized
            .checked_add(1)
            .expect("invariant: parallel map initialized count is representable");

        // SAFETY: the constructor grants this writer exclusive access to the
        // range and the bounds check above keeps `index` inside it.
        unsafe {
            self.values.add(index).write(MaybeUninit::new(value));
        }
        self.initialized = next_initialized;
    }

    /// Publish ownership of a completely initialized chunk range.
    pub(super) fn finish(mut self) -> Range<usize> {
        let expected = self.range.end - self.range.start;
        assert_eq!(
            self.initialized, expected,
            "invariant: parallel map chunk initialized every output slot"
        );
        self.armed = false;
        self.range.clone()
    }
}

impl<T> Drop for ChunkWriter<T> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        for offset in 0..self.initialized {
            let index = self.range.start + offset;
            // SAFETY: `initialized` advances only after `push` writes the slot,
            // and this writer retains exclusive ownership until `finish`.
            unsafe {
                ptr::drop_in_place(self.values.add(index).cast::<T>());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChunkWriter, MapOutput};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Tracked<'a>(&'a AtomicUsize);

    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn completed_chunks_transfer_the_original_allocation() {
        let mut output = MapOutput::new(5, 2);
        let values = output.values_ptr();
        let completed = output.completed_ptr();

        // SAFETY: the two writers receive disjoint ranges within the five-slot
        // allocation, and both writers finish before the allocation is moved.
        let first = unsafe {
            let mut writer = ChunkWriter::new(values, 0..2);
            writer.push(10);
            writer.push(11);
            writer.finish()
        };
        // SAFETY: this range is disjoint from the first writer and remains
        // within the same live five-slot allocation.
        let second = unsafe {
            let mut writer = ChunkWriter::new(values, 2..5);
            writer.push(12);
            writer.push(13);
            writer.push(14);
            writer.finish()
        };
        // SAFETY: both indices belong to the live two-slot completion array;
        // each range is published exactly once after its writer finishes.
        unsafe {
            completed.write(Some(first));
            completed.add(1).write(Some(second));
        }

        let result = output.into_vec();
        assert_eq!(result.as_ptr(), values.cast::<i32>());
        assert_eq!(result, [10, 11, 12, 13, 14]);
    }

    #[test]
    fn unfinished_writer_drops_only_its_initialized_prefix() {
        let drops = AtomicUsize::new(0);
        let mut output = MapOutput::new(4, 1);

        // SAFETY: the writer owns the complete live four-slot allocation and
        // is dropped before the allocation itself.
        let mut writer = unsafe { ChunkWriter::new(output.values_ptr(), 0..4) };
        writer.push(Tracked(&drops));
        writer.push(Tracked(&drops));
        drop(writer);

        assert_eq!(drops.load(Ordering::Relaxed), 2);
        drop(output);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn zero_sized_outputs_retain_their_logical_length() {
        let mut output = MapOutput::new(3, 1);
        let values = output.values_ptr();
        let completed = output.completed_ptr();

        // SAFETY: the writer exclusively owns the full logical range of the
        // live zero-sized allocation and initializes every logical slot.
        let range = unsafe {
            let mut writer = ChunkWriter::new(values, 0..3);
            writer.push(());
            writer.push(());
            writer.push(());
            writer.finish()
        };
        // SAFETY: index zero is the sole live completion slot.
        unsafe {
            completed.write(Some(range));
        }

        assert_eq!(output.into_vec(), [(), (), ()]);
    }
}
