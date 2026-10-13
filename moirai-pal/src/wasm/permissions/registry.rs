use crate::permission_state::PermissionState;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};
use web_sys::Document;

#[derive(Default)]
pub(super) struct Registry {
    pub(super) state: RefCell<PermissionState>,
}

struct Current {
    document: Document,
    registry: Weak<Registry>,
}

thread_local! {
    // One current Document key; unsettled workers retain only Registry state,
    // never application elements or the document. Realm destruction releases
    // the browser's remaining callbacks and this WASM instance.
    static CURRENT: RefCell<Option<Current>> = const { RefCell::new(None) };
}

pub(super) fn for_document(document: &Document) -> Rc<Registry> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if let Some(existing) = current.as_ref()
            && existing.document == *document
            && let Some(registry) = existing.registry.upgrade()
        {
            return registry;
        }
        let registry = Rc::new(Registry::default());
        *current = Some(Current {
            document: document.clone(),
            registry: Rc::downgrade(&registry),
        });
        registry
    })
}
