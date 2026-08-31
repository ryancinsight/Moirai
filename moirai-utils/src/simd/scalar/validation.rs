#[inline]
pub(super) fn scalar_matrix_shape<const N: usize>(left: &[f32], right: &[f32], result: &mut [f32]) {
    assert!(N != 0, "matrix dimension must be non-zero");
    let expected = N.checked_mul(N).expect("matrix dimension overflow");
    assert_eq!(left.len(), expected, "left matrix size must equal N * N");
    assert_eq!(right.len(), expected, "right matrix size must equal N * N");
    assert_eq!(
        result.len(),
        expected,
        "result matrix size must equal N * N"
    );
}
