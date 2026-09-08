//! Browser text-control selection and composition metadata.

use super::{WebElement, WebEvent};
pub use crate::text_model::{TextSelection, TextSelectionDirection};
use crate::text_validation::{event_data, input_type, locale, text_value};
use std::io;
use wasm_bindgen::JsCast;
use web_sys::{CompositionEvent, HtmlInputElement, HtmlTextAreaElement, InputEvent};

/// Metadata emitted by one browser InputEvent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextInputMetadata {
    value: String,
    data: Option<String>,
    input_type: String,
    composing: bool,
    selection: TextSelection,
}

impl TextInputMetadata {
    /// Returns the bounded current value of the text control.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the optional inserted or deleted text reported by the event.
    #[must_use]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    /// Returns the browser input-operation name.
    #[must_use]
    pub fn input_type(&self) -> &str {
        &self.input_type
    }

    /// Returns whether the browser reports an active composition.
    #[must_use]
    pub const fn is_composing(&self) -> bool {
        self.composing
    }

    /// Returns the value selection after the input operation.
    #[must_use]
    pub const fn selection(&self) -> TextSelection {
        self.selection
    }
}

/// Metadata emitted by one browser CompositionEvent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionMetadata {
    data: Option<String>,
    locale: String,
}

impl CompositionMetadata {
    /// Returns the optional preedit or committed composition text.
    #[must_use]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    /// Returns the browser-reported input locale.
    #[must_use]
    pub fn locale(&self) -> &str {
        &self.locale
    }
}

impl WebElement {
    /// Reads a bounded value from an HTML input or textarea control.
    ///
    /// # Errors
    /// Returns io::ErrorKind::InvalidInput when the browser value exceeds the
    /// provider text bound.
    pub fn text_value(&self) -> io::Result<Option<String>> {
        let value = self
            .element
            .dyn_ref::<HtmlInputElement>()
            .map(HtmlInputElement::value)
            .or_else(|| {
                self.element
                    .dyn_ref::<HtmlTextAreaElement>()
                    .map(HtmlTextAreaElement::value)
            });
        value.map(text_value).transpose()
    }

    /// Reads the selection of an HTML input or textarea control.
    ///
    /// # Errors
    /// Returns io::ErrorKind::InvalidInput when the browser rejects a
    /// selection read or reports an inverted range.
    pub fn text_selection(&self) -> io::Result<Option<TextSelection>> {
        if let Some(input) = self.element.dyn_ref::<HtmlInputElement>() {
            return read_input_selection(input);
        }
        if let Some(textarea) = self.element.dyn_ref::<HtmlTextAreaElement>() {
            return read_textarea_selection(textarea);
        }
        Ok(None)
    }

    /// Restores a selection on an HTML input or textarea control.
    ///
    /// # Errors
    /// Returns io::ErrorKind::InvalidInput when the target is not a text
    /// control, the selection direction is not supported, or the browser
    /// rejects the range.
    pub fn set_text_selection(&self, selection: TextSelection) -> io::Result<()> {
        let Some(value) = self.text_value()? else {
            return Err(invalid_selection_error());
        };
        if self.text_selection()?.is_none() {
            return Err(invalid_selection_error());
        }
        let value_units = value.encode_utf16().count();
        let value_units = u32::try_from(value_units).map_err(|_| invalid_selection_error())?;
        if selection.end() > value_units {
            return Err(selection_range_error());
        }
        let direction = match selection.direction() {
            TextSelectionDirection::Forward => "forward",
            TextSelectionDirection::Backward => "backward",
            TextSelectionDirection::None => "none",
            TextSelectionDirection::Other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Browser does not accept an unknown text selection direction",
                ));
            }
        };
        if let Some(input) = self.element.dyn_ref::<HtmlInputElement>() {
            return input
                .set_selection_range_with_direction(selection.start(), selection.end(), direction)
                .map_err(|_| invalid_selection_error());
        }
        if let Some(textarea) = self.element.dyn_ref::<HtmlTextAreaElement>() {
            return textarea
                .set_selection_range_with_direction(selection.start(), selection.end(), direction)
                .map_err(|_| invalid_selection_error());
        }
        Err(invalid_selection_error())
    }
}

impl WebEvent {
    /// Reads bounded text and selection metadata from a browser InputEvent.
    ///
    /// Non-input events and unsupported targets return Ok(None). Browser
    /// metadata is copied only after the provider bounds are checked.
    ///
    /// # Errors
    /// Returns io::ErrorKind::InvalidInput when the browser rejects a
    /// selection read or a metadata bound is exceeded.
    pub fn text_input_metadata(&self) -> io::Result<Option<TextInputMetadata>> {
        let Some(input) = self.event.dyn_ref::<InputEvent>() else {
            return Ok(None);
        };
        let Some(target) = self.target() else {
            return Ok(None);
        };
        let Some(value) = target.text_value()? else {
            return Ok(None);
        };
        let Some(selection) = target.text_selection()? else {
            return Ok(None);
        };
        Ok(Some(TextInputMetadata {
            value,
            data: event_data(input.data())?,
            input_type: input_type(input.input_type())?,
            composing: input.is_composing(),
            selection,
        }))
    }

    /// Reads bounded preedit metadata from a browser CompositionEvent.
    ///
    /// Non-composition events return Ok(None). Composition data is optional
    /// because browsers may omit it for a start or cancellation transition.
    ///
    /// # Errors
    /// Returns io::ErrorKind::InvalidInput when event data or locale exceeds
    /// the provider bounds.
    pub fn composition_metadata(&self) -> io::Result<Option<CompositionMetadata>> {
        let Some(composition) = self.event.dyn_ref::<CompositionEvent>() else {
            return Ok(None);
        };
        Ok(Some(CompositionMetadata {
            data: event_data(composition.data())?,
            locale: locale(composition.locale())?,
        }))
    }
}

fn read_input_selection(input: &HtmlInputElement) -> io::Result<Option<TextSelection>> {
    let Some(start) = input
        .selection_start()
        .map_err(|_| invalid_selection_error())?
    else {
        return Ok(None);
    };
    let Some(end) = input
        .selection_end()
        .map_err(|_| invalid_selection_error())?
    else {
        return Ok(None);
    };
    TextSelection::new(
        start,
        end,
        TextSelectionDirection::from_browser(
            input
                .selection_direction()
                .map_err(|_| invalid_selection_error())?
                .as_deref(),
        ),
    )
    .map(Some)
}

fn read_textarea_selection(textarea: &HtmlTextAreaElement) -> io::Result<Option<TextSelection>> {
    let Some(start) = textarea
        .selection_start()
        .map_err(|_| invalid_selection_error())?
    else {
        return Ok(None);
    };
    let Some(end) = textarea
        .selection_end()
        .map_err(|_| invalid_selection_error())?
    else {
        return Ok(None);
    };
    TextSelection::new(
        start,
        end,
        TextSelectionDirection::from_browser(
            textarea
                .selection_direction()
                .map_err(|_| invalid_selection_error())?
                .as_deref(),
        ),
    )
    .map(Some)
}

fn invalid_selection_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "Browser rejected text selection access",
    )
}

fn selection_range_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "Text selection exceeds the control value",
    )
}
