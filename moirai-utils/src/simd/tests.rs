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
fn add_falls_back_on_unaligned_input() {
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
