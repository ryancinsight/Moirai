//! Closed fixture wire, shared by the compiled browser and real native owner.

use serde::{Deserialize, Deserializer, Serialize};

/// Maximum complete key notices; callbacks after the bound retain a real error.
pub(super) const MAX_KEY_RECORDS: u32 = 64;
/// Reserve the longest successful Closed notice, using the full counter width.
/// Failure strings are not truncated; a larger actual failure must still fit
/// the aggregate remaining capacity or fail publication.
pub(super) const SUCCESSFUL_CLOSED_BYTES: usize = r#"{"kind":"closed","key_events":4294967295,"publication_error":null,"bridge_retained":0,"dom_active":0,"dom_retained":0}"#.len();

#[derive(Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Command {
    Close {},
}

#[derive(Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Notice {
    Ready {},
    Key {
        sequence: u32,
        keyboard: KeyboardSnapshot,
    },
    Closed {
        key_events: u32,
        #[serde(deserialize_with = "required_option")]
        publication_error: Option<String>,
        bridge_retained: usize,
        dom_active: usize,
        dom_retained: usize,
    },
}

fn required_option<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

/// Every field is an actual provider observation, not accelerator policy.
#[derive(Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Six independent observed KeyboardEvent flags form a 2^6 Cartesian domain, verified by exact named-field codec tests"
)]
pub(super) struct KeyboardSnapshot {
    pub(super) key: String,
    pub(super) code: String,
    pub(super) ctrl: bool,
    pub(super) alt: bool,
    pub(super) shift: bool,
    pub(super) meta: bool,
    pub(super) repeat: bool,
    pub(super) trusted: bool,
}

#[cfg(target_arch = "wasm32")]
impl From<moirai_pal::wasm::KeyboardMetadata> for KeyboardSnapshot {
    fn from(value: moirai_pal::wasm::KeyboardMetadata) -> Self {
        let modifiers = value.modifiers();
        Self {
            key: value.key().to_owned(),
            code: value.code().to_owned(),
            ctrl: modifiers.ctrl(),
            alt: modifiers.alt(),
            shift: modifiers.shift(),
            meta: modifiers.meta(),
            repeat: value.is_repeat(),
            trusted: value.is_trusted(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, KeyboardSnapshot, MAX_KEY_RECORDS, Notice, SUCCESSFUL_CLOSED_BYTES};
    use tracing_subscriber::fmt::format::FmtSpan;

    fn trace() -> tracing::subscriber::DefaultGuard {
        tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_test_writer()
                .with_span_events(FmtSpan::FULL)
                .finish(),
        )
    }

    #[test]
    fn independent_keyboard_flags_preserve_named_wire_values() {
        let _trace = trace();
        let _test = tracing::info_span!("keyboard_fixture_flags").entered();
        for mask in 0_u8..(1 << 6) {
            let _vector = tracing::info_span!("vector", mask).entered();
            let (flags, notice) = tracing::info_span!("arrange").in_scope(|| {
                let flags: [bool; 6] = std::array::from_fn(|bit| mask & (1 << bit) != 0);
                let [ctrl, alt, shift, meta, repeat, trusted] = flags;
                (
                    flags,
                    Notice::Key {
                        sequence: u32::from(mask) + 1,
                        keyboard: KeyboardSnapshot {
                            key: "患者-🦀-\"\\\n".into(),
                            code: "KeyQ".into(),
                            ctrl,
                            alt,
                            shift,
                            meta,
                            repeat,
                            trusted,
                        },
                    },
                )
            });
            let (wire, decoded) = tracing::info_span!("act").in_scope(|| {
                let encoded = serde_json::to_string(&notice).expect("actual fixture encoder");
                let wire: serde_json::Value = serde_json::from_str(&encoded).expect("wire object");
                let decoded: Notice =
                    serde_json::from_str(&encoded).expect("actual fixture decoder");
                (wire, decoded)
            });
            tracing::info_span!("assert").in_scope(|| {
                for (name, flag) in ["ctrl", "alt", "shift", "meta", "repeat", "trusted"]
                    .into_iter()
                    .zip(flags)
                {
                    assert_eq!(wire["keyboard"][name], flag, "{name}");
                }
                assert_eq!(wire["sequence"], u32::from(mask) + 1);
                assert_eq!(wire["keyboard"]["key"], "患者-🦀-\"\\\n");
                let Notice::Key { keyboard, .. } = &decoded else {
                    panic!("decoded actual key notice has another variant");
                };
                assert_eq!(
                    [
                        keyboard.ctrl,
                        keyboard.alt,
                        keyboard.shift,
                        keyboard.meta,
                        keyboard.repeat,
                        keyboard.trusted
                    ],
                    flags
                );
                assert_eq!(decoded, notice);
            });
        }
    }

    #[test]
    fn lifecycle_notices_preserve_values_and_successful_reserve() {
        let _trace = trace();
        let _test = tracing::info_span!("keyboard_fixture_lifecycle").entered();
        let notices = tracing::info_span!("arrange").in_scope(|| {
            [
                Notice::Ready {},
                Notice::Closed {
                    key_events: u32::MAX,
                    publication_error: None,
                    bridge_retained: 0,
                    dom_active: 0,
                    dom_retained: 0,
                },
                Notice::Closed {
                    key_events: MAX_KEY_RECORDS,
                    publication_error: Some("actual error text 患者-🦀-\"\\\n".into()),
                    bridge_retained: 1,
                    dom_active: 2,
                    dom_retained: 3,
                },
            ]
        });
        for notice in notices {
            let (encoded, decoded) = tracing::info_span!("act").in_scope(|| {
                let encoded = serde_json::to_string(&notice).expect("notice encoder");
                let decoded: Notice = serde_json::from_str(&encoded).expect("notice decoder");
                (encoded, decoded)
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(decoded, notice);
                if matches!(
                    notice,
                    Notice::Closed {
                        publication_error: None,
                        ..
                    }
                ) {
                    assert_eq!(encoded.len(), SUCCESSFUL_CLOSED_BYTES);
                }
                let value: serde_json::Value = serde_json::from_str(&encoded).expect("wire notice");
                if let Notice::Closed {
                    bridge_retained,
                    dom_active,
                    dom_retained,
                    ..
                } = notice
                {
                    assert_eq!(value["bridge_retained"], bridge_retained);
                    assert_eq!(value["dom_active"], dom_active);
                    assert_eq!(value["dom_retained"], dom_retained);
                }
            });
        }
        let command = serde_json::to_string(&Command::Close {}).expect("actual close encoder");
        assert_eq!(command, r#"{"kind":"close"}"#);
        assert_eq!(
            serde_json::from_str::<Command>(&command).expect("close decoder"),
            Command::Close {}
        );
    }

    #[test]
    fn original_closed_schema_rejects_unknown_duplicate_and_missing_fields() {
        let _trace = trace();
        let _test = tracing::info_span!("keyboard_fixture_closed_wire").entered();
        let commands = tracing::info_span!("arrange").in_scope(|| {
            [
                (r#"{"kind":"close","unknown":1}"#, "unknown field `unknown`"),
                (
                    r#"{"kind":"close","kind":"close"}"#,
                    "duplicate field `kind`",
                ),
                (r#"{"kind":"other"}"#, "unknown variant `other`"),
            ]
        });
        for (original, expected) in commands {
            let error = tracing::info_span!("act").in_scope(|| {
                serde_json::from_str::<Command>(original).expect_err("closed command rejection")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert!(error.is_data());
                assert!(
                    error.to_string().contains(expected),
                    "{error:?}: {expected}"
                );
                // Internally tagged Serde variants deserialize buffered content;
                // custom Data errors can carry no original JSON location (0,0).
                assert!(
                    (error.line(), error.column()) == (0, 0)
                        || (error.line() == 1
                            && error.column() > 0
                            && error.column() <= original.len()),
                    "actual source location: {error:?}"
                );
            });
        }
        let originals = [
            (r#"{"kind":"ready","extra":true}"#, "unknown field `extra`"),
            (
                r#"{"kind":"key","sequence":1,"keyboard":{"key":"D","code":"KeyD","ctrl":false,"alt":true,"shift":true,"meta":false,"repeat":false,"trusted":true,"extra":true}}"#,
                "unknown field `extra`",
            ),
            (
                r#"{"kind":"key","sequence":1,"keyboard":{"key":"D","code":"KeyD","ctrl":false,"alt":true,"shift":true,"meta":false,"repeat":false,"trusted":true,"ctrl":true}}"#,
                "duplicate field `ctrl`",
            ),
            (
                r#"{"kind":"closed","key_events":1,"publication_error":null,"bridge_retained":0,"dom_retained":0}"#,
                "missing field `dom_active`",
            ),
            (
                r#"{"kind":"closed","key_events":1,"bridge_retained":0,"dom_active":0,"dom_retained":0}"#,
                "missing field `publication_error`",
            ),
        ];
        for (original, expected) in originals {
            let error = tracing::info_span!("act").in_scope(|| {
                serde_json::from_str::<Notice>(original).expect_err("closed notice rejection")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert!(error.is_data());
                assert!(
                    error.to_string().contains(expected),
                    "{error:?}: {expected}"
                );
                // Internally tagged Serde variants deserialize buffered content;
                // custom Data errors can carry no original JSON location (0,0).
                assert!(
                    (error.line(), error.column()) == (0, 0)
                        || (error.line() == 1
                            && error.column() > 0
                            && error.column() <= original.len()),
                    "actual source location: {error:?}"
                );
            });
        }
    }
}
