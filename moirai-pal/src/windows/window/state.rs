//! Bounded callback state for a native window.

use std::collections::VecDeque;
use std::io;

use super::config::{MAX_WINDOW_EVENTS, allocation_error};
use super::event::{CompositionPhase, ModifierState, WindowEvent};
use super::input::update_modifier;

#[derive(Debug)]
pub(super) struct PresentedFrame {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: Vec<u32>,
}

#[derive(Debug)]
pub(super) struct WindowState {
    pub(super) events: VecDeque<WindowEvent>,
    pub(super) frame: Option<PresentedFrame>,
    pending_high_surrogate: Option<u16>,
    pub(super) composition_active: bool,
    pub(super) modifiers: ModifierState,
    pub(super) overflowed: bool,
    pub(super) error: Option<io::Error>,
}

impl WindowState {
    pub(super) fn new() -> io::Result<Self> {
        let mut events = VecDeque::new();
        events
            .try_reserve(MAX_WINDOW_EVENTS)
            .map_err(|_| allocation_error())?;
        Ok(Self {
            events,
            frame: None,
            pending_high_surrogate: None,
            composition_active: false,
            modifiers: ModifierState::NONE,
            overflowed: false,
            error: None,
        })
    }

    pub(super) fn push(&mut self, event: WindowEvent) {
        if self.events.len() >= MAX_WINDOW_EVENTS {
            self.overflowed = true;
        } else {
            self.events.push_back(event);
        }
    }

    pub(super) fn push_composition(&mut self, phase: CompositionPhase, text: String) {
        self.composition_active =
            matches!(phase, CompositionPhase::Started | CompositionPhase::Updated);
        self.push(WindowEvent::TextComposition { phase, text });
    }

    pub(super) fn record_error(&mut self, error: io::Error) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    pub(super) fn update_modifier(&mut self, virtual_key: u32, pressed: bool) {
        self.modifiers = update_modifier(self.modifiers, virtual_key, pressed);
    }

    pub(super) fn clear_modifiers(&mut self) {
        self.modifiers = ModifierState::NONE;
    }

    pub(super) fn push_text_unit(&mut self, unit: u16) {
        match (self.pending_high_surrogate.take(), unit) {
            (Some(high), low @ 0xdc00..=0xdfff) => {
                let scalar =
                    0x1_0000 + ((u32::from(high) - 0xd800) << 10) + (u32::from(low) - 0xdc00);
                if let Some(character) = char::from_u32(scalar) {
                    self.push(WindowEvent::TextInput { character });
                } else {
                    self.push(WindowEvent::TextInput {
                        character: '\u{fffd}',
                    });
                }
            }
            (Some(_), _) => {
                self.push(WindowEvent::TextInput {
                    character: '\u{fffd}',
                });
                self.push_text_unit(unit);
            }
            (None, high @ 0xd800..=0xdbff) => {
                self.pending_high_surrogate = Some(high);
            }
            (None, 0xdc00..=0xdfff) => {
                self.push(WindowEvent::TextInput {
                    character: '\u{fffd}',
                });
            }
            (None, unit) => {
                if let Some(character) = char::from_u32(u32::from(unit)) {
                    self.push(WindowEvent::TextInput { character });
                }
            }
        }
    }

    pub(super) fn finish_text(&mut self) {
        if self.pending_high_surrogate.take().is_some() {
            self.push(WindowEvent::TextInput {
                character: '\u{fffd}',
            });
        }
    }
}

pub(super) fn decode_composition(units: &[u16]) -> io::Result<String> {
    if units.len() > super::config::MAX_COMPOSITION_UNITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition exceeds the bounded UTF-16 limit",
        ));
    }
    String::from_utf16(units).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition contains invalid UTF-16",
        )
    })
}
