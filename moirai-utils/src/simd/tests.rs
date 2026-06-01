use super::*;

#[test]
fn detects_platform_capabilities() {
    let _ = has_avx2_support();
    let _ = has_neon_support();
    let _ = has_native_vector_path::<f32>();
}

#[test]
fn add_preserves_values() {
    let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let right = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut result = [0.0; 8];

    add(&left, &right, &mut result);

    assert_eq!(result, [9.0; 8]);
}

#[test]
fn add_handles_short_unaligned_input() {
    let left = [1.0, 2.0, 3.0];
    let right = [4.0, 5.0, 6.0];
    let mut result = [0.0; 3];

    add(&left, &right, &mut result);

    assert_eq!(result, [5.0, 7.0, 9.0]);
}

#[test]
fn mul_preserves_values() {
    let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let right = [2.0; 8];
    let mut result = [0.0; 8];

    mul(&left, &right, &mut result);

    assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
}

#[test]
fn dot_preserves_values() {
    let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let right = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let result = dot(&left, &right);

    assert_eq!(result, 120.0);
}

#[test]
fn matrix_mul_square_preserves_identity_and_order() {
    let left = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let right = [
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 4.0,
    ];
    let expected = [
        5.0, 2.0, 3.0, 30.0, 13.0, 6.0, 7.0, 70.0, 21.0, 10.0, 11.0, 110.0, 29.0, 14.0, 15.0, 150.0,
    ];

    let mut result = [0.0; 16];
    matrix_mul_square::<f32, 4>(&left, &identity, &mut result);
    assert_eq!(result, left);

    matrix_mul_square::<f32, 4>(&left, &right, &mut result);
    assert_eq!(result, expected);

    let mut reversed = [0.0; 16];
    matrix_mul_square::<f32, 4>(&right, &left, &mut reversed);
    assert_ne!(result, reversed);
}

#[test]
fn statistics_preserve_values() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let total = sum(&data);
    let average = mean(&data);
    let spread = variance(&data);

    assert_eq!(total, 36.0);
    assert_eq!(average, 4.5);
    assert_eq!(spread, 5.25);
}

#[test]
fn integer_add_and_dot_preserve_native_type() {
    let left = [1_u64, 2, 3, 4];
    let right = [5_u64, 6, 7, 8];
    let mut result = [0_u64; 4];

    add(&left, &right, &mut result);
    let product_sum = dot(&left, &right);

    assert_eq!(result, [6, 8, 10, 12]);
    assert_eq!(product_sum, 70);
}

#[test]
fn unaligned_lengths_preserve_values() {
    for len in [3, 7, 9, 11, 15, 17, 23, 31, 33] {
        let left: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let right: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();

        let mut result_add = vec![0.0; len];
        add(&left, &right, &mut result_add);
        let expected_add: Vec<f32> = left.iter().zip(right.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(result_add, expected_add, "add mismatch at len {len}");

        let mut result_mul = vec![0.0; len];
        mul(&left, &right, &mut result_mul);
        let expected_mul: Vec<f32> = left.iter().zip(right.iter()).map(|(x, y)| x * y).collect();
        assert_eq!(result_mul, expected_mul, "mul mismatch at len {len}");

        let result_dot = dot(&left, &right);
        let expected_dot: f32 = left.iter().zip(right.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result_dot - expected_dot).abs() < 1e-4,
            "dot mismatch at len {len}"
        );

        let result_sum = sum(&left);
        let expected_sum: f32 = left.iter().copied().sum();
        assert_eq!(result_sum, expected_sum, "sum mismatch at len {len}");

        let result_var = variance(&left);
        let mean_val = mean(&left);
        let expected_var = left
            .iter()
            .copied()
            .map(|value| {
                let diff = value - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / len as f32;
        assert!(
            (result_var - expected_var).abs() < 1e-4,
            "variance mismatch at len {len}: result={result_var}, expected={expected_var}"
        );
    }
}

#[test]
fn unaligned_vector_prefix_records_vector_dispatch_when_available() {
    let counter = crate::global_simd_counter();
    counter.reset();

    let len = 17;
    let left: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let right: Vec<f32> = (0..len).map(|i| (i * 2) as f32).collect();
    let mut result = vec![0.0; len];

    add(&left, &right, &mut result);

    let expected: Vec<f32> = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left + right)
        .collect();
    assert_eq!(result, expected);

    let (vectorized_ops, scalar_ops, vectorized_elements, scalar_elements) = counter.get_stats();
    if has_native_vector_path::<f32>() {
        assert_eq!(vectorized_ops, 1);
        assert_eq!(scalar_ops, 0);
        assert_eq!(vectorized_elements, len);
        assert_eq!(scalar_elements, 0);
    } else {
        assert_eq!(vectorized_ops, 0);
        assert_eq!(scalar_ops, 1);
        assert_eq!(vectorized_elements, 0);
        assert_eq!(scalar_elements, len);
    }
}
