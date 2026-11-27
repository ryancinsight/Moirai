use wasm_bindgen_test::wasm_bindgen_test;

use super::{error, interface};

#[wasm_bindgen_test]
fn actual_browser_exposes_the_unconstructible_webidl_interface() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("geolocation_webidl_interface").entered();
    let window = web_sys::window().expect("actual browser window");
    let geolocation = tracing::info_span!("act")
        .in_scope(|| interface(&window))
        .expect("Navigator.geolocation getter")
        .expect("actual browser geolocation interface");
    tracing::info_span!("assert").in_scope(|| {
        for method in ["watchPosition", "clearWatch"] {
            assert_eq!(
                error::method(geolocation.as_ref(), method),
                Ok(true),
                "browser geolocation must expose {method}"
            );
        }
    });
}
