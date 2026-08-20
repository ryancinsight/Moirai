//! Parallel reduction over a large dataset using work-stealing tasks.
//!
//! Divide a 1 M-element array into chunks, sum each chunk in a separate
//! task, and reduce the partial sums.  The work-stealing scheduler
//! balances the load across all available CPU cores automatically.

use moirai::Moirai;

const N: usize = 1_000_000;
const CHUNKS: usize = 8;

fn main() {
    let runtime = Moirai::new().expect("failed to create Moirai runtime");

    // Build the data.
    let data: Vec<u64> = (0..N as u64).collect();

    // Divide into CHUNKS equal pieces and sum each chunk in a task.
    let chunk_size = N / CHUNKS;
    let handles: Vec<_> = data
        .chunks(chunk_size)
        .map(|chunk| {
            // Copy the slice into a Vec so the closure is 'static + Send.
            let owned: Vec<u64> = chunk.to_vec();
            runtime.spawn_fn(move || owned.iter().copied().sum::<u64>())
        })
        .collect();

    let total: u64 = handles
        .into_iter()
        .map(|h| h.join().expect("join").expect("task"))
        .sum();

    let expected: u64 = (N as u64 * (N as u64 - 1)) / 2; // Gauss formula
    println!("parallel sum of 0..{N}: {total}");
    println!("expected              : {expected}");
    assert_eq!(
        total, expected,
        "parallel reduction must equal the sequential sum"
    );

    // Throughput note.
    println!(
        "processed {} elements across {CHUNKS} tasks ({} per task)",
        N, chunk_size,
    );
    println!("parallel-reduce assertion passed");
}
