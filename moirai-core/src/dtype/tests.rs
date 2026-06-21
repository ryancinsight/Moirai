use super::*;

#[test]
fn test_integer_dtype_operations() {
    let a: i32 = 10;
    let b: i32 = 3;

    assert_eq!(a.checked_add(b), Some(13));
    assert_eq!(a.checked_sub(b), Some(7));
    assert_eq!(a.checked_mul(b), Some(30));
    assert_eq!(a.checked_div(b), Some(3));

    assert_eq!(a.saturating_add(i32::MAX), i32::MAX);
    assert_eq!(a.abs(), 10);
    assert!(a.is_positive());
    assert!(!a.is_negative());
    assert!(!a.is_zero());

    assert!(a.is_even());
    assert!(!b.is_even());
    assert!(!a.is_odd());
    assert!(b.is_odd());
}

#[test]
fn test_float_dtype_operations() {
    let a: f64 = 10.5;
    let b: f64 = 3.2;

    assert!(a.checked_add(b).unwrap().approx_eq(13.7));
    assert!(a.checked_sub(b).unwrap().approx_eq(7.3));
    assert!(a.checked_mul(b).unwrap().approx_eq(33.6));
    assert!(a.checked_div(b).unwrap().approx_eq(3.28125));

    assert!(!a.is_nan());
    assert!(a.is_finite());
    assert!(!a.is_infinite());

    assert!(a.floor().approx_eq(10.0));
    assert!(a.ceil().approx_eq(11.0));
    assert!(a.round().approx_eq(11.0));
    assert!(a.trunc().approx_eq(10.0));

    assert!(a.sqrt().approx_eq(3.240_370_349_203_93));
}

#[test]
fn test_overflow_safety() {
    let max_val = i32::MAX;
    assert_eq!(max_val.checked_add(1), None);
    assert_eq!(max_val.saturating_add(1), i32::MAX);

    let min_val = i32::MIN;
    assert_eq!(min_val.checked_sub(1), None);
    assert_eq!(min_val.saturating_sub(1), i32::MIN);
}

#[test]
fn test_float_precision() {
    let a: f64 = 0.1 + 0.2;
    let b: f64 = 0.3;

    assert!(a.approx_eq(b));

    let c: f64 = 1.0;
    let d: f64 = 2.0;
    assert!(!c.approx_eq(d));
}

#[test]
fn test_compute_context() {
    let ctx = ComputeContext::<f64>::with_epsilon();
    assert_eq!(ctx.tolerance.unwrap().to_bits(), f64::EPSILON.to_bits());
    assert_eq!(ctx.max_iterations, 1000);
    assert!(ctx.check_overflow);

    let ctx2 = ComputeContext::<i32>::default();
    assert!(ctx2.tolerance.is_none());
    assert_eq!(ctx2.max_iterations, 1000);
    assert!(ctx2.check_overflow);
}
