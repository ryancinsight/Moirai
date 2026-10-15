//! A real MTA UI Automation client; its COM objects never cross threads.

use std::{collections::BTreeMap, time::Instant};
use windows::{
    Win32::{
        Foundation::HWND,
        System::{
            Com::{
                CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
                CoUninitialize,
            },
            Variant::VARIANT,
        },
        UI::Accessibility::{
            CUIAutomation, IUIAutomation, IUIAutomationElement, IUIAutomationValuePattern,
            TreeScope_Descendants, UIA_AutomationIdPropertyId, UIA_ValuePatternId,
        },
    },
    core::Interface,
};

const FIELDS: [&str; 4] = ["reference", "weight", "concentration", "dose"];
const CASES: [(&str, &str, &str, &str); 4] = [
    ("患者-🦀-one", "64", "2", "0.25"),
    ("患者-🦀-two", "80", "4", "0.5"),
    ("患者-invalid", "80", "0", "0.5"),
    ("患者-🦀-three", "96", "8", "0.125"),
];

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Completion {
    schema: u8,
    window: String,
    replacements: usize,
    preserved_checks: usize,
    final_values: BTreeMap<String, Vec<u16>>,
}

impl Completion {
    pub(super) fn verify(&self, handle: usize) {
        assert_eq!(self.schema, 1);
        assert_eq!(self.window, handle.to_string());
        assert_eq!(self.replacements, CASES.len() * FIELDS.len());
        assert_eq!(self.preserved_checks, self.replacements * FIELDS.len());
        assert_eq!(self.final_values.len(), FIELDS.len());
        let last = CASES.last().expect("declared nonempty input domain");
        for (id, value) in FIELDS.into_iter().zip([last.0, last.1, last.2, last.3]) {
            assert_eq!(
                self.final_values.get(id).expect("actual final field"),
                &value.encode_utf16().collect::<Vec<_>>()
            );
        }
    }
}

struct Apartment;
impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: constructed only after this thread's successful COM entry;
        // every client interface is declared later and drops before the guard.
        unsafe { CoUninitialize() };
    }
}

pub(super) fn replace_fields(handle: usize, deadline: Instant) -> Completion {
    // SAFETY: this worker owns its MTA and all UIA client interfaces.
    unsafe { CoInitializeEx(None, COINIT_MULTITHREADED).ok() }.expect("UIA MTA");
    let _apartment = Apartment;
    let automation: IUIAutomation =
        unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER) }
            .expect("actual system UIA client");
    let root =
        unsafe { automation.ElementFromHandle(HWND(handle as *mut _)) }.expect("native UIA root");
    let mut values = BTreeMap::new();
    for id in FIELDS {
        let element = unique(&automation, &root, id, deadline);
        values.insert(id, current_value(&element));
    }
    let mut replacements = 0_usize;
    let mut preserved_checks = 0_usize;
    for (reference, weight, concentration, dose) in CASES {
        for (id, expected) in [
            ("reference", reference),
            ("weight", weight),
            ("concentration", concentration),
            ("dose", dose),
        ] {
            let _case = tracing::info_span!("uia_replacement", id, expected).entered();
            let element = unique(&automation, &root, id, deadline);
            unsafe { element.SetFocus() }.expect("actual UIA SetFocus");
            // Observe completion under the same deadline; SetFocus does not
            // make a concurrent UIA property snapshot into an event barrier.
            let focus_started = Instant::now();
            let mut observations = 0_usize;
            loop {
                observations = observations
                    .checked_add(1)
                    .expect("bounded UIA focus observations");
                if unsafe { element.CurrentHasKeyboardFocus() }
                    .expect("actual UIA focus")
                    .as_bool()
                {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "actual UIA focus did not reach {id}"
                );
                std::thread::yield_now();
            }
            tracing::info!(id, apartment = "MTA", observations, elapsed_ns = %focus_started.elapsed().as_nanos(), "actual UIA focus observed");
            let pattern: IUIAutomationValuePattern =
                unsafe { element.GetCurrentPattern(UIA_ValuePatternId) }
                    .expect("actual ValuePattern")
                    .cast()
                    .expect("actual typed ValuePattern");
            super::input::select_all(handle);
            let expected_units: Vec<_> = expected.encode_utf16().collect();
            super::input::insert_unicode(handle, expected);
            loop {
                let actual = unsafe { pattern.CurrentValue() }.expect("actual UIA value");
                if &*actual == expected_units.as_slice() {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "UIA {id} replacement: expected {expected:?}, actual units {:?}",
                    &*actual
                );
                std::thread::yield_now();
            }
            values.insert(id, expected_units);
            for (other_id, expected_units) in &values {
                let other = unique(&automation, &root, other_id, deadline);
                assert_eq!(
                    current_value(&other),
                    *expected_units,
                    "replacement of {id} preserves {other_id}"
                );
                preserved_checks = preserved_checks
                    .checked_add(1)
                    .expect("declared field domain");
            }
            replacements = replacements
                .checked_add(1)
                .expect("declared replacement domain");
        }
    }
    Completion {
        schema: 1,
        window: handle.to_string(),
        replacements,
        preserved_checks,
        final_values: values
            .into_iter()
            .map(|(id, value)| (id.to_owned(), value))
            .collect(),
    }
}

fn unique(
    automation: &IUIAutomation,
    root: &IUIAutomationElement,
    id: &str,
    deadline: Instant,
) -> IUIAutomationElement {
    // SAFETY: all interfaces belong to this MTA, the VARIANT is valid for the
    // synchronous call, and the returned collection owns each interface.
    let condition = unsafe {
        automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, &VARIANT::from(id))
    }
    .expect("actual UIA id condition");
    loop {
        let matches =
            unsafe { root.FindAll(TreeScope_Descendants, &condition) }.expect("actual UIA lookup");
        let count = unsafe { matches.Length() }.expect("actual UIA count");
        assert!(count <= 1, "ambiguous native UIA control {id}");
        if count == 1 {
            return unsafe { matches.GetElement(0) }.expect("unique native UIA element");
        }
        assert!(Instant::now() < deadline, "native UIA control absent: {id}");
        std::thread::yield_now();
    }
}

fn current_value(element: &IUIAutomationElement) -> Vec<u16> {
    // SAFETY: the client owns these COM interfaces on this MTA. BSTR preserves
    // the actual UTF-16 units; no string normalization enters the assertion.
    let pattern: IUIAutomationValuePattern =
        unsafe { element.GetCurrentPattern(UIA_ValuePatternId) }
            .expect("actual ValuePattern")
            .cast()
            .expect("actual typed ValuePattern");
    unsafe { pattern.CurrentValue() }
        .expect("actual UIA value")
        .to_vec()
}
