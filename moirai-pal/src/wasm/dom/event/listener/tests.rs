//! Real DOM registration, delivery, teardown and native TypeError retention.

use std::{cell::Cell, io::ErrorKind, rc::Rc};

use js_sys::Reflect;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::{EventTarget, HtmlElement};

use super::{WebEventListener, listen};

#[wasm_bindgen_test]
fn actual_dom_listener_owners_close_and_retain_failed_cleanup() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("dom_listener_ownership").entered();
    let (element, calls) = tracing::info_span!("arrange").in_scope(|| {
        let element = web_sys::window()
            .expect("actual browser Window")
            .document()
            .expect("actual Document")
            .create_element("button")
            .expect("actual DOM element")
            .dyn_into::<HtmlElement>()
            .expect("actual HTML button");
        (element, Rc::new(Cell::new(0)))
    });
    let target: EventTarget = element.clone().unchecked_into();
    assert_eq!(
        (
            WebEventListener::active_count(),
            WebEventListener::retained_cleanup_count()
        ),
        (0, 0)
    );
    let calls_first = Rc::clone(&calls);
    let mut first = listen(target.clone(), "click", move |event| {
        assert!(!event.is_trusted());
        calls_first.set(calls_first.get() + 4);
    })
    .expect("first actual listener");
    let calls_second = Rc::clone(&calls);
    let second = listen(target.clone(), "click", move |event| {
        // Programmatic HTML click is actual browser delivery with untrusted
        // provenance. This unit test makes no physical input acceptance claim.
        assert!(!event.is_trusted());
        calls_second.set(calls_second.get() + 1);
    })
    .expect("second actual listener");
    tracing::info_span!("act").in_scope(|| {
        assert_eq!(WebEventListener::active_count(), 2);
        assert_eq!(first.close().expect("explicit native detach"), ());
        element.click();
        drop(second);
        element.click();
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(calls.get(), 1);
        assert_eq!(
            (
                WebEventListener::active_count(),
                WebEventListener::retained_cleanup_count()
            ),
            (0, 0)
        );
        assert_eq!(first.close().expect("idempotent detached owner"), ());
    });
    let retained_calls = Rc::clone(&calls);
    let mut retained = listen(target.clone(), "click", move |event| {
        assert!(!event.is_trusted());
        retained_calls.set(retained_calls.get() + 1)
    })
    .expect("real retained listener");
    let original = Reflect::get(target.as_ref(), &JsValue::from_str("removeEventListener"))
        .expect("native removal method");
    // Replacing this actual DOM method with a noncallable value causes the
    // browser's genuine TypeError. The registered callback/target remain real;
    // no script function, fake bridge, event or computed result is substituted.
    assert!(
        Reflect::set(
            target.as_ref(),
            &JsValue::from_str("removeEventListener"),
            &JsValue::FALSE
        )
        .expect("actual DOM property mutation")
    );
    let error = retained
        .close()
        .expect_err("native noncallable method TypeError");
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(
        error
            .to_string()
            .contains("DOM removeEventListener: TypeError:"),
        "{error}"
    );
    let before_drop = listen(target.clone(), "click", |event| {
        assert!(!event.is_trusted());
    })
    .err()
    .expect("first explicit close failure already closes admission");
    assert_eq!(before_drop.kind(), ErrorKind::WouldBlock);
    assert_eq!(WebEventListener::active_count(), 1);
    assert_eq!(WebEventListener::retained_cleanup_count(), 0);
    drop(retained);
    assert_eq!(
        (
            WebEventListener::active_count(),
            WebEventListener::retained_cleanup_count()
        ),
        (0, 1)
    );
    let error = listen(target.clone(), "click", |event| {
        assert!(!event.is_trusted())
    })
    .err()
    .expect("realm admission refuses failed cleanup");
    assert_eq!(error.kind(), ErrorKind::WouldBlock);
    assert_eq!(
        error.to_string(),
        "DOM realm retains failed event listener cleanup"
    );
    assert!(
        Reflect::set(
            target.as_ref(),
            &JsValue::from_str("removeEventListener"),
            &original
        )
        .expect("restore actual method")
    );
    element.click();
    assert_eq!(
        calls.get(),
        2,
        "retained attached callback remains valid after failed Drop"
    );
    assert_eq!(WebEventListener::retained_cleanup_count(), 1);
}
