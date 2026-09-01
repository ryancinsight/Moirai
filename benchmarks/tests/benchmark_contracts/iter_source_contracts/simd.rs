#[test]
fn iterator_simd_surface_uses_generic_scalar_contract() {
    let source = read_benchmark("../moirai-iter/src/simd_iter.rs");
    let benchmark = read_benchmark("benches/iter_simd_comparison.rs");
    let manifest = read_benchmark("Cargo.toml");
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");
    let backup = benchmark_path("../moirai-iter/src/simd_iter_backup.rs");

    for required in [
        "pub trait SimdScalar",
        "mod sealed",
        "pub struct SimdSliceIter<'a, T>",
        "impl<'a, T: SimdScalar> SimdSliceIter<'a, T>",
        "pub fn add_slice(self, other: &'a [T]) -> Vec<T>",
        "pub fn scale(self, scalar: T) -> Vec<T>",
        "pub fn dot(self, other: &'a [T]) -> T",
        "pub struct CacheFriendlyIterator<T>",
        "(CACHE_LINE_SIZE / scalar_size).max(1)",
        "pub fn reduce<T, F, R>(data: &[T], identity: R, op: F) -> R",
        "pub fn filter<T, P>(data: Vec<T>, predicate: P) -> Vec<T>",
        "generic_slice_addition_preserves_values",
        "generic_slice_scale_preserves_native_precision_values",
        "generic_slice_dot_preserves_values",
        "cache_friendly_iterator_processes_large_elements",
        "simd_ops_reduce_and_filter_are_value_semantic",
    ] {
        assert!(
            source.contains(required),
            "SIMD iterator source must retain generic scalar marker {required}"
        );
    }

    for prohibited in [
        "SimdF32Iterator",
        "simd_add",
        "simd_multiply",
        "simd_dot_product",
        "simd_parallel_reduce",
        "pub const AVX2_F32_WIDTH",
        "pub const SSE2_F32_WIDTH",
        "CACHE_FRIENDLY_CHUNK_SIZE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>()",
        "For now",
        "placeholder",
        "Real implementation",
    ] {
        assert!(
            !source.contains(prohibited),
            "SIMD iterator source must not retain non-generic or placeholder marker {prohibited}"
        );
    }

    assert!(
        !backup.exists(),
        "stale SIMD backup source must not remain in the repository"
    );

    for required in [
        "name = \"iter_simd_comparison\"",
        "SimdSliceIter::new(left).add_slice(right)",
        "SimdSliceIter::new(left).dot(right)",
        "assert_eq!(generic_add(&left, &right), scalar_add(&left, &right))",
        "assert_eq!(generic_dot(&left, &right), scalar_dot(&left, &right))",
        "iter_simd_generic_add",
        "iter_simd_generic_dot",
    ] {
        assert!(
            benchmark.contains(required) || manifest.contains(required),
            "SIMD iterator benchmark must retain executable marker {required}"
        );
    }

    for required in [
        "Iterator SIMD surface is generic",
        "iter_simd_comparison",
        "SimdSliceIter<T>",
    ] {
        assert!(
            audit.contains(required),
            "Rayon/Tokio audit must retain SIMD cleanup marker {required}"
        );
    }
}
