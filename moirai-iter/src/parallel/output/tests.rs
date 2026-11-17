use super::{ChunkWriter, MapOutput, output_chunk_range};
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

    // `catch_unwind` over a closure that always panics has `!` in its `Ok`
    // position, so `Err(payload)` is irrefutable and the `else` arm it used
    // to carry was unreachable.
    let payload =
        panic.expect_err("invariant: the simulated mapper panic must unwind through the writer");
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
