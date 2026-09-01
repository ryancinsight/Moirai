#[test]
fn sorting_slice_extension_is_value_semantic_and_benchmarked() {
    let audit = read_benchmark("../docs/rayon_adapter_surface_audit.md");
    let comparison_report = read_benchmark("../docs/moirai_rayon_tokio_comparison.md");
    let sorting_source = read_benchmark("../moirai-iter/src/parallel/sorting.rs");
    let parallel_root = read_benchmark("../moirai-iter/src/parallel.rs");
    let lib_root = read_benchmark("../moirai-iter/src/lib.rs");
    let sorting_benchmark = read_benchmark("benches/sorting_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");

    for required in [
        "mod sorting;",
        "pub use sorting::ParallelSliceMut;",
        "ParallelSliceMut",
    ] {
        assert!(
            parallel_root.contains(required) || lib_root.contains(required),
            "parallel sorting boundary must retain exported marker {required}"
        );
    }

    for required in [
        "pub trait ParallelSliceMut<T: Send>",
        "fn par_sort(&mut self)",
        "fn par_sort_by<F>(&mut self, compare: F)",
        "fn par_sort_by_key<K, F>(&mut self, f: F)",
        "fn par_sort_unstable(&mut self)",
        "fn par_sort_unstable_by<F>(&mut self, compare: F)",
        "fn par_sort_unstable_by_key<K, F>(&mut self, f: F)",
        "const STABLE_SEQUENTIAL_THRESHOLD: usize = 2048;",
        "const UNSTABLE_SEQUENTIAL_THRESHOLD: usize = 16_384;",
        // The fork-join runs on the scheduler scope, not a crate-owned pool
        // (ADR-022), and splits only while a sub-slice is worth another lane.
        "par_merge_sort_impl(executor, self, &compare, grain)",
        "par_sort_unstable_by_impl(executor, self, &compare, grain)",
        "fn fork_join_halves",
        "executor.scope::<SyncTask, _>",
        "fn fork_grain",
        "test_sorting_empty_and_single",
        "test_sorting_large_random",
        "test_sorting_stability",
        "test_panic_safety_no_double_drop",
    ] {
        assert!(
            sorting_source.contains(required),
            "parallel sorting source must retain marker {required}"
        );
    }

    for prohibited in [
        "sorting adapters | no parallel sorting adapters",
        "sorting adapters remain unsupported",
    ] {
        assert!(
            !audit.contains(prohibited),
            "Rayon adapter audit must not retain obsolete sorting unsupported marker {prohibited}"
        );
        assert!(
            !comparison_report.contains(prohibited),
            "Rayon/Tokio comparison report must not retain obsolete sorting unsupported marker {prohibited}"
        );
    }

    for required in [
        "Slice extension boundary",
        "ParallelSliceMut",
        "sorting_comparison",
        "ParallelSliceMut` benchmark",
    ] {
        assert!(
            audit.contains(required),
            "Rayon adapter audit must retain sorting boundary marker {required}"
        );
    }

    for required in [
        "Rayon Adapter Surface Boundary",
        "inspect",
        "panic_fuse",
        "chunks",
        "partition",
        "ParallelSliceMut` for stable and unstable slice sorting",
        "Stable slice sort",
        "Unstable slice sort",
        "Full Rayon ecosystem parity is incomplete",
    ] {
        assert!(
            comparison_report.contains(required),
            "Rayon/Tokio comparison report must retain current adapter marker {required}"
        );
    }

    for required in [
        "name = \"sorting_comparison\"",
        "MoiraiParallelSliceMut::par_sort",
        "RayonParallelSliceMut::par_sort",
        "MoiraiParallelSliceMut::par_sort_unstable",
        "RayonParallelSliceMut::par_sort_unstable",
        "assert_eq!(moirai_stable, rayon_stable)",
        "assert_eq!(moirai_unstable, rayon_unstable)",
        "parallel_sorting_stable",
        "parallel_sorting_unstable",
    ] {
        assert!(
            sorting_benchmark.contains(required) || benchmark_manifest.contains(required),
            "sorting benchmark must retain comparison marker {required}"
        );
    }
}
