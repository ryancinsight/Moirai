//! ECMAScript Number decimal oracles at rounding and exponent boundaries.

use wasm_bindgen_test::wasm_bindgen_test;

use super::WebNumber;

#[wasm_bindgen_test]
fn actual_number_decimal_matches_ecmascript_oracles() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("web_number_decimal_contract").entered();
    let cases = tracing::info_span!("arrange").in_scope(|| {
        [
            (0.0, "0"),
            (-0.0, "0"),
            (1.25, "1.25"),
            (-125.5, "-125.5"),
            (1e-7, "1e-7"),
            (1e-6, "0.000001"),
            (1e20, "100000000000000000000"),
            (1e21, "1e+21"),
            (9_007_199_254_740_991.0, "9007199254740991"),
            (9_007_199_254_740_992.0, "9007199254740992"),
            (f64::from_bits(1), "5e-324"),
            (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
            (f64::MAX, "1.7976931348623157e+308"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (f64::NAN, "NaN"),
        ]
    });
    for (value, expected) in cases {
        let actual = tracing::info_span!("act").in_scope(|| WebNumber::decimal(value));
        tracing::info_span!("assert")
            .in_scope(|| assert_eq!(actual.expect("actual Number.toString"), expected));
    }
}
