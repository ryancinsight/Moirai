//! A real MTA UI Automation client; its COM objects never cross threads.

use super::{ObservationMode, input::Observations};
use serde_json::json;
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

mod focus;

pub(super) const FIELDS: [&str; 4] = ["reference", "weight", "concentration", "dose"];
pub(super) const CASES: [(&str, &str, &str, &str); 4] = [
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
    mode: String,
    replacements: usize,
    preserved_checks: usize,
    parent_entries: usize,
    nonfirst_parent_entries: usize,
    final_values: BTreeMap<String, Vec<u16>>,
}

impl Completion {
    pub(super) fn verify(&self, handle: usize, mode: ObservationMode) {
        assert_eq!(self.schema, 3);
        assert_eq!(self.window, handle.to_string());
        assert_eq!(self.mode, mode.as_str());
        assert_eq!(self.replacements, CASES.len() * FIELDS.len());
        assert_eq!(self.preserved_checks, self.replacements * FIELDS.len());
        assert_eq!(self.parent_entries, self.replacements);
        assert_eq!(
            self.nonfirst_parent_entries,
            CASES.len() * (FIELDS.len() - 1)
        );
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

pub(super) fn replace_fields(
    handle: usize,
    mode: ObservationMode,
    deadline: Instant,
) -> Completion {
    let mut observations = Observations::new(mode, 0);
    let mut probe = super::renderer::Emitter::new(handle, mode);
    // SAFETY: this worker owns its MTA and all UIA client interfaces.
    observations
        .operation("CoInitializeEx", || unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()
        })
        .expect("UIA MTA");
    let _apartment = Apartment;
    let automation: IUIAutomation = observations
        .operation("CoCreateInstance", || unsafe {
            CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER)
        })
        .expect("actual system UIA client");
    let root = observations
        .operation("ElementFromHandle", || unsafe {
            automation.ElementFromHandle(HWND(handle as *mut _))
        })
        .expect("native UIA root");
    let mut values = BTreeMap::new();
    for id in FIELDS {
        let element = unique(&automation, &root, id, deadline, &mut observations);
        values.insert(id, current_value(&element, &mut observations));
    }
    drop(observations);
    let mut replacements = 0_usize;
    let mut preserved_checks = 0_usize;
    let mut parent_entries = 0_usize;
    let mut nonfirst_parent_entries = 0_usize;
    for (reference, weight, concentration, dose) in CASES {
        for (field_index, (id, expected)) in [
            ("reference", reference),
            ("weight", weight),
            ("concentration", concentration),
            ("dose", dose),
        ]
        .into_iter()
        .enumerate()
        {
            let _case = tracing::info_span!("uia_replacement", id, expected).entered();
            let mut observations = Observations::new(mode, replacements + 1);
            let element = unique(&automation, &root, id, deadline, &mut observations);
            focus::enter(&element, handle, deadline, &mut observations);
            parent_entries = parent_entries
                .checked_add(1)
                .expect("declared entry domain");
            if field_index != 0 {
                nonfirst_parent_entries = nonfirst_parent_entries
                    .checked_add(1)
                    .expect("declared non-first field entry domain");
            }
            let pattern = value_pattern(&element, &mut observations);
            super::input::select_all(handle, &mut observations);
            let expected_units: Vec<_> = expected.encode_utf16().collect();
            super::input::insert_unicode(handle, expected, &mut observations);
            await_value(
                &pattern,
                &expected_units,
                deadline,
                &mut observations,
                &mut probe,
            );
            values.insert(id, expected_units);
            for (other_id, expected_units) in &values {
                let other = unique(&automation, &root, other_id, deadline, &mut observations);
                assert_eq!(
                    current_value(&other, &mut observations),
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
        schema: 3,
        window: handle.to_string(),
        mode: mode.as_str().to_owned(),
        replacements,
        preserved_checks,
        parent_entries,
        nonfirst_parent_entries,
        final_values: values
            .into_iter()
            .map(|(id, value)| (id.to_owned(), value))
            .collect(),
    }
}

fn unique(
    automation: &IUIAutomation,
    root: &IUIAutomationElement,
    id: &'static str,
    deadline: Instant,
    observations: &mut Observations,
) -> IUIAutomationElement {
    observations.field(id);
    // SAFETY: all interfaces belong to this MTA, the VARIANT is valid for the
    // synchronous call, and the returned collection owns each interface.
    let condition = observations
        .operation("CreatePropertyCondition", || unsafe {
            automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, &VARIANT::from(id))
        })
        .expect("actual UIA id condition");
    loop {
        let matches = observations
            .operation("FindAll", || unsafe {
                root.FindAll(TreeScope_Descendants, &condition)
            })
            .expect("actual UIA lookup");
        let count = observations
            .operation("Length", || unsafe { matches.Length() })
            .expect("actual UIA count");
        observations.record(json!({"kind":"lookup_count", "count":count}));
        assert!(count <= 1, "ambiguous native UIA control {id}");
        if count == 1 {
            return observations
                .operation("GetElement", || unsafe { matches.GetElement(0) })
                .expect("unique native UIA element");
        }
        assert!(Instant::now() < deadline, "native UIA control absent: {id}");
        std::thread::yield_now();
    }
}

fn value_pattern(
    element: &IUIAutomationElement,
    observations: &mut Observations,
) -> IUIAutomationValuePattern {
    // SAFETY: the client owns these COM interfaces on this MTA. BSTR preserves
    // the actual UTF-16 units; no string normalization enters the assertion.
    let pattern = observations
        .operation("GetCurrentPattern", || unsafe {
            element.GetCurrentPattern(UIA_ValuePatternId)
        })
        .expect("actual ValuePattern");
    observations
        .operation("ValuePattern_cast", || pattern.cast())
        .expect("actual typed ValuePattern")
}

fn current_value(element: &IUIAutomationElement, observations: &mut Observations) -> Vec<u16> {
    let pattern = value_pattern(element, observations);
    let value = observations
        .operation("CurrentValue", || unsafe { pattern.CurrentValue() })
        .expect("actual UIA value")
        .to_vec();
    observations.record(json!({"kind":"value_snapshot", "units":value}));
    value
}

fn await_value(
    pattern: &IUIAutomationValuePattern,
    expected: &[u16],
    deadline: Instant,
    observations: &mut Observations,
    probe: &mut super::renderer::Emitter,
) {
    let mut queries = 0_usize;
    let mut previous = None;
    loop {
        queries = queries
            .checked_add(1)
            .expect("bounded UIA value observations");
        let actual = observations
            .operation("CurrentValue", || unsafe { pattern.CurrentValue() })
            .expect("actual UIA value")
            .to_vec();
        if previous.as_ref() != Some(&actual) {
            observations.record(json!({"kind":"value_snapshot", "query":queries,"units":actual}));
            previous = Some(actual.clone());
        }
        if actual == expected {
            observations
                .record(json!({"kind":"value_completed", "queries":queries,"units":actual}));
            break;
        }
        probe.mismatch(observations, expected, &actual, queries, deadline);
        assert!(
            Instant::now() < deadline,
            "UIA replacement: expected units {expected:?}, actual units {actual:?}"
        );
        std::thread::yield_now();
    }
}
