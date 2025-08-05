//! Tests for iterator fixes

use moirai_iter::base::{SlidingWindow, ChunksExact};
use moirai_iter::advanced_iterators::{AdvancedIteratorExt, ScanRef};

#[test]
fn test_scan_ref_lifetime() {
    // Test that scan_ref works with non-static data
    let data = vec![1, 2, 3, 4, 5];
    let mut sum = 0;
    
    let results: Vec<i32> = data.iter()
        .scan_ref(0, |state, &x| {
            *state += x;
            sum = *state;
            Some(*state)
        })
        .collect();
    
    assert_eq!(results, vec![1, 3, 6, 10, 15]);
    assert_eq!(sum, 15);
}

#[test]
fn test_sliding_window_double_ended() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut windows = SlidingWindow::new(&data, 3);
    
    // Test forward iteration
    assert_eq!(windows.next(), Some(&[1, 2, 3][..]));
    assert_eq!(windows.next(), Some(&[2, 3, 4][..]));
    
    // Test backward iteration
    assert_eq!(windows.next_back(), Some(&[6, 7, 8][..]));
    assert_eq!(windows.next_back(), Some(&[5, 6, 7][..]));
    
    // Test that they meet in the middle correctly
    assert_eq!(windows.next(), Some(&[3, 4, 5][..]));
    assert_eq!(windows.next_back(), Some(&[4, 5, 6][..]));
    
    // No more windows
    assert_eq!(windows.next(), None);
    assert_eq!(windows.next_back(), None);
}

#[test]
fn test_sliding_window_exact_size() {
    let data = vec![1, 2, 3, 4, 5];
    let windows = SlidingWindow::new(&data, 2);
    
    assert_eq!(windows.len(), 4); // [1,2], [2,3], [3,4], [4,5]
    assert_eq!(windows.size_hint(), (4, Some(4)));
}

#[test]
fn test_chunks_exact_remainder() {
    let data = vec![1, 2, 3, 4, 5, 6, 7];
    let chunks = ChunksExact::new(&data, 3);
    
    // Check remainder
    assert_eq!(chunks.remainder(), &[7]);
    
    // Check chunks
    let collected: Vec<_> = chunks.collect();
    assert_eq!(collected, vec![&[1, 2, 3][..], &[4, 5, 6][..]]);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_async_execution_no_blocking() {
    use moirai_iter::{moirai_iter, ParallelContext};
    use std::time::Instant;
    
    let data: Vec<i32> = (1..=100).collect();
    let start = Instant::now();
    
    // This should not block the async runtime
    let result = moirai_iter(data)
        .map(|x| x * 2)
        .reduce(|a, b| a + b)
        .await;
    
    let elapsed = start.elapsed();
    
    assert_eq!(result, Some(10100)); // sum of 2..=200
    
    // Should complete quickly without blocking
    assert!(elapsed.as_millis() < 100, "Operation took too long, might be blocking");
}

#[test]
fn test_partition_ref() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    let (evens, odds): (Vec<_>, Vec<_>) = data.into_iter()
        .partition_ref(|&x| x % 2 == 0)
        .partition();
    
    assert_eq!(evens, vec![2, 4, 6, 8, 10]);
    assert_eq!(odds, vec![1, 3, 5, 7, 9]);
}