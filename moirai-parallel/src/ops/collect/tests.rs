use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicUsize, Ordering},
};

use melinoe::MelinoeCell;
use moirai_executor::{SyncTask, global};

use crate::{
    Parallel, map_collect_index_with, map_collect_mut_with, map_collect_with,
    melinoe_ext::par_partition_map_with_policy,
};

#[derive(Default)]
struct Calls {
    attempted: AtomicUsize,
    dropped: AtomicUsize,
}

impl Calls {
    fn result(&self) -> ResultValue<'_> {
        assert_ne!(
            self.attempted.fetch_add(1, Ordering::Relaxed),
            1,
            "mapper panic"
        );
        ResultValue(&self.dropped)
    }
}

struct ResultValue<'a>(&'a AtomicUsize);

impl Drop for ResultValue<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn check_unwind(collect: impl Fn(&Calls) + Send + Sync) {
    // A nested indexed region runs on the current lane. This ensures one
    // mapper call returns a value before the second call panics.
    global()
        .for_each_indexed::<SyncTask, _>(1, |_| {
            let calls = Calls::default();
            let error = catch_unwind(AssertUnwindSafe(|| collect(&calls)))
                .expect_err("mapper panic must reach the collector's caller");
            assert!(
                error
                    .downcast_ref::<String>()
                    .is_some_and(|message| message.contains("SpawnFailed(Panicked)"))
            );
            assert_eq!(calls.attempted.load(Ordering::Relaxed), 2);
            assert_eq!(calls.dropped.load(Ordering::Relaxed), 1);
        })
        .expect("the outer indexed region must complete");
}

#[test]
fn plain_map_drops_initialized_output_on_unwind() {
    check_unwind(|calls| {
        drop(map_collect_with::<Parallel, _, _, _>(&[10, 20, 30], |_| {
            calls.result()
        }));
    });
}

#[test]
fn indexed_map_drops_initialized_output_on_unwind() {
    check_unwind(|calls| {
        drop(map_collect_index_with::<Parallel, _, _>(3, |_| {
            calls.result()
        }));
    });
}

#[test]
fn mutable_map_drops_initialized_output_on_unwind() {
    check_unwind(|calls| {
        drop(map_collect_mut_with::<Parallel, _, _, _>(
            &mut [10, 20, 30],
            |_, value| {
                *value += 1;
                calls.result()
            },
        ));
    });
}

#[test]
fn partition_map_drops_initialized_output_on_unwind() {
    check_unwind(|calls| {
        let mut source = [10, 20, 30];
        let cells = MelinoeCell::from_mut_slice(&mut source);
        drop(par_partition_map_with_policy::<Parallel, _, _, _>(
            cells,
            1,
            |_, _| calls.result(),
        ));
    });
}
