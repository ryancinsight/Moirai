//! Run a bounded nested parallel workload outside the libtest harness.
//!
//! The process shape is useful when investigating failures that appear only in
//! a test binary. Pass the number of nested passes as the first argument; the
//! default is 300 and the hard bound is 20,000.

use std::{
    env,
    error::Error,
    io::{Error as IoError, ErrorKind},
    num::NonZeroUsize,
};

use moirai_iter::{IntoParallelIterator, ParallelIterator};

const DEFAULT_ITERATIONS: usize = 300;
const MAX_ITERATIONS: usize = 20_000;
const OUTER_ITEMS: usize = 1_025;
const INNER_ITEMS: usize = 1_025;

#[derive(Clone, Copy)]
struct IterationCount(NonZeroUsize);

impl IterationCount {
    fn get(self) -> usize {
        self.0.get()
    }
}

fn parse_iterations() -> Result<IterationCount, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let raw = args
        .next()
        .unwrap_or_else(|| DEFAULT_ITERATIONS.to_string());
    if args.next().is_some() {
        return Err(IoError::new(ErrorKind::InvalidInput, "expected one iteration count").into());
    }

    let value = raw.parse::<usize>()?;
    let count = NonZeroUsize::new(value).ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "iteration count must be greater than zero",
        )
    })?;
    if count.get() > MAX_ITERATIONS {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("iteration count exceeds the {MAX_ITERATIONS} pass bound"),
        )
        .into());
    }
    Ok(IterationCount(count))
}

fn nested_pass() {
    let expected_inner: usize = (0..INNER_ITEMS).sum();
    let results: Vec<usize> = (0..OUTER_ITEMS)
        .into_par_iter()
        .map(|x| {
            let inner_sum = match (0..INNER_ITEMS).into_par_iter().reduce(|a, b| a + b) {
                Some(sum) => sum,
                None => panic!("invariant: non-empty inner range reduces to a value"),
            };
            inner_sum + x
        })
        .collect();

    assert_eq!(results.len(), OUTER_ITEMS);
    for (x, result) in results.iter().copied().enumerate() {
        assert_eq!(result, expected_inner + x);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let iterations = parse_iterations()?.get();
    for _ in 0..iterations {
        nested_pass();
    }
    println!("completed {iterations} nested passes");
    Ok(())
}
