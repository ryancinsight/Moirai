//! Safe, owned browser DOM handles for Atlas applications.

mod canvas;
mod clipboard;
mod content_box;
mod document;
mod event;
mod file_drop;
mod gpu_canvas;
mod keyboard;
mod page;
mod text;

pub use self::canvas::{CanvasSize, RgbaFrame, WebCanvas};
pub use self::clipboard::WebClipboard;
pub use self::event::listener::WebEventListener;
pub use self::event::{
    PointerMetadata, PointerModifiers, PointerType, WebEvent, WheelDeltaMode, WheelMetadata,
};
pub use self::file_drop::{BrowserFiles, DropFiles, DropMetadata, DroppedFile, DroppedFileAccess};
pub use self::gpu_canvas::WebGpuCanvas;
pub use self::keyboard::KeyboardMetadata;
pub use self::page::WebLocation;
pub use self::text::{
    CompositionMetadata, TextInputMetadata, TextSelection, TextSelectionDirection,
};

use std::io;

pub use crate::content_box::ContentBoxPoint;
use wasm_bindgen::JsCast;
use web_sys::{
    Document, Element, HtmlButtonElement, HtmlDialogElement, HtmlElement, HtmlInputElement,
    HtmlSelectElement, HtmlTextAreaElement,
};

/// A browser document obtained from the current window.
#[derive(Clone)]
pub struct WebDocument {
    document: Document,
}

/// An owned browser element handle.
#[derive(Clone)]
pub struct WebElement {
    element: Element,
}

/// An element extent measured in CSS pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementSize {
    width: f64,
    height: f64,
}

impl ElementSize {
    /// Returns the rendered width in CSS pixels.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    /// Returns the rendered height in CSS pixels.
    #[must_use]
    pub const fn height(self) -> f64 {
        self.height
    }
}

impl WebElement {
    /// Returns the element's identifier, or an empty string when it has none.
    #[must_use]
    pub fn id(&self) -> String {
        self.element.id()
    }

    /// Replaces the element's HTML contents.
    ///
    /// The caller owns the markup string and must only pass trusted
    /// application-authored content. Untrusted values belong in
    /// [`Self::set_text`].
    pub fn set_inner_html(&self, markup: &str) {
        self.element.set_inner_html(markup);
    }

    /// Replaces the element's text content without interpreting markup.
    pub fn set_text(&self, text: &str) {
        self.element.set_text_content(Some(text));
    }

    /// Sets one DOM attribute.
    ///
    /// # Errors
    /// Returns the browser's DOM validation error as [`io::ErrorKind::InvalidInput`].
    pub fn set_attribute(&self, name: &str, value: &str) -> io::Result<()> {
        self.element
            .set_attribute(name, value)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid DOM attribute"))
    }

    /// Sets one inline CSS property through the browser's CSS object model.
    ///
    /// Property-by-property mutation avoids replacing the element's complete
    /// `style` attribute and remains usable when a content security policy
    /// rejects application-authored inline style declarations.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when this is not an HTML
    /// element or setting the property throws a browser exception.
    pub fn set_style_property(&self, name: &str, value: &str) -> io::Result<()> {
        let element = self.element.dyn_ref::<HtmlElement>().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "DOM element has no CSS style")
        })?;
        element.style().set_property(name, value).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser rejected CSS style property",
            )
        })
    }

    /// Returns the element's rendered border-box extent in fractional CSS pixels.
    ///
    /// The dimensions include padding and borders and reflect CSS transforms,
    /// matching the browser's `getBoundingClientRect` semantics.
    #[must_use]
    pub fn bounding_size(&self) -> ElementSize {
        let bounds = self.element.get_bounding_client_rect();
        ElementSize {
            width: bounds.width(),
            height: bounds.height(),
        }
    }

    /// Maps a viewport client point into the untransformed local content box.
    ///
    /// The returned point carries the content extent measured in the same
    /// layout read as the coordinate. Borders and padding are excluded. The
    /// mapping composes this element's and its ancestors' two-dimensional
    /// affine CSS transforms. The bounding rectangle supplies only the
    /// transformed box translation; its transformed extent is never treated as
    /// a local content size.
    ///
    /// # Errors
    /// Returns a typed error for unavailable or invalid CSS geometry,
    /// three-dimensional or singular transforms, motion paths, or CSS zoom.
    /// Closed shadow roots cannot expose their composed transform ancestry;
    /// callers embedding a canvas there must keep transforming wrappers inside
    /// the inspectable root.
    pub fn content_box_point(&self, client_x: f64, client_y: f64) -> io::Result<ContentBoxPoint> {
        content_box::measure_point(&self.element, client_x, client_y)
    }

    /// Returns the disabled state of a button, input, or select control.
    #[must_use]
    pub fn disabled(&self) -> Option<bool> {
        self.element
            .dyn_ref::<HtmlButtonElement>()
            .map(HtmlButtonElement::disabled)
            .or_else(|| {
                self.element
                    .dyn_ref::<HtmlInputElement>()
                    .map(HtmlInputElement::disabled)
            })
            .or_else(|| {
                self.element
                    .dyn_ref::<HtmlSelectElement>()
                    .map(HtmlSelectElement::disabled)
            })
    }

    /// Sets the disabled state of a button, input, or select control.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when this element is not a
    /// disableable form control.
    pub fn set_disabled(&self, disabled: bool) -> io::Result<()> {
        if let Some(button) = self.element.dyn_ref::<HtmlButtonElement>() {
            button.set_disabled(disabled);
            return Ok(());
        }
        if let Some(input) = self.element.dyn_ref::<HtmlInputElement>() {
            input.set_disabled(disabled);
            return Ok(());
        }
        if let Some(select) = self.element.dyn_ref::<HtmlSelectElement>() {
            select.set_disabled(disabled);
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "DOM element is not a disableable form control",
        ))
    }

    /// Returns whether this HTML dialog is open.
    #[must_use]
    pub fn dialog_open(&self) -> Option<bool> {
        self.element
            .dyn_ref::<HtmlDialogElement>()
            .map(HtmlDialogElement::open)
    }

    /// Opens this HTML dialog as a modal dialog.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when this element is not a
    /// dialog or when the browser rejects the modal transition.
    pub fn show_modal(&self) -> io::Result<()> {
        let dialog = self.element.dyn_ref::<HtmlDialogElement>().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "DOM element is not a dialog")
        })?;
        dialog.show_modal().map(|_| ()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "Browser rejected modal dialog")
        })
    }

    /// Closes this HTML dialog.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when this element is not a
    /// dialog.
    pub fn close_dialog(&self) -> io::Result<()> {
        let dialog = self.element.dyn_ref::<HtmlDialogElement>().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "DOM element is not a dialog")
        })?;
        dialog.close();
        Ok(())
    }

    /// Moves browser focus to this HTML element.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when this element is not an
    /// HTML element or when the browser rejects the focus request.
    pub fn focus(&self) -> io::Result<()> {
        let element = self.element.dyn_ref::<HtmlElement>().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "DOM element cannot receive focus",
            )
        })?;
        element.focus().map(|_| ()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser rejected focus request",
            )
        })
    }

    /// Captures subsequent pointer events for this element.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser rejects the
    /// pointer identifier or capture request.
    pub fn set_pointer_capture(&self, pointer_id: i32) -> io::Result<()> {
        self.element.set_pointer_capture(pointer_id).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser rejected pointer capture",
            )
        })
    }

    /// Releases a pointer previously captured by this element.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser rejects the
    /// pointer identifier or release request.
    pub fn release_pointer_capture(&self, pointer_id: i32) -> io::Result<()> {
        self.element
            .release_pointer_capture(pointer_id)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Browser rejected pointer release",
                )
            })
    }

    /// Returns whether this element currently captures a pointer identifier.
    #[must_use]
    pub fn has_pointer_capture(&self, pointer_id: i32) -> bool {
        self.element.has_pointer_capture(pointer_id)
    }

    /// Appends a child and returns no detached handle.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser rejects the
    /// parent/child relationship.
    pub fn append_child(&self, child: &Self) -> io::Result<()> {
        self.element
            .append_child(&child.element)
            .map(|_| ())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid DOM child"))
    }

    /// Reads the value of a browser input or select element.
    #[must_use]
    pub fn value(&self) -> Option<String> {
        self.element
            .dyn_ref::<HtmlInputElement>()
            .map(HtmlInputElement::value)
            .or_else(|| {
                self.element
                    .dyn_ref::<HtmlSelectElement>()
                    .map(HtmlSelectElement::value)
            })
    }

    /// Reads the checked state of a browser input element.
    ///
    /// Returns [`None`] when this element is not an input. Checkbox and radio
    /// controls expose their current state through the same browser property;
    /// other input kinds return their browser-defined unchecked state.
    #[must_use]
    pub fn checked(&self) -> Option<bool> {
        self.element
            .dyn_ref::<HtmlInputElement>()
            .map(HtmlInputElement::checked)
    }

    /// Replaces the value of a browser input, textarea or select element.
    ///
    /// A select element selects the option whose value matches. Browsers
    /// clear the selection for a value no option carries; that value is
    /// rejected instead and the previous selection is kept.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the element is not a
    /// value-bearing control or a select has no option with `value`.
    pub fn set_value(&self, value: &str) -> io::Result<()> {
        if let Some(input) = self.element.dyn_ref::<HtmlInputElement>() {
            input.set_value(value);
            return Ok(());
        }
        if let Some(textarea) = self.element.dyn_ref::<HtmlTextAreaElement>() {
            textarea.set_value(value);
            return Ok(());
        }
        if let Some(select) = self.element.dyn_ref::<HtmlSelectElement>() {
            let previous = select.value();
            select.set_value(value);
            if select.value() == value {
                return Ok(());
            }
            select.set_value(&previous);
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Select element has no option with the requested value",
            ));
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "DOM element is not an input, textarea or select",
        ))
    }

    /// Registers a callback whose lifetime is tied to the returned listener.
    ///
    /// Explicit [`WebEventListener::close`] reports browser removal failures.
    /// Drop attempts removal before releasing the closure; a failed removal
    /// retains a counted realm owner and prevents further listener admission.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the event name is invalid
    /// or the browser refuses the listener, and [`io::ErrorKind::WouldBlock`]
    /// when this realm retains a failed listener cleanup.
    pub fn add_event_listener<F>(
        &self,
        event_name: &str,
        callback: F,
    ) -> io::Result<WebEventListener>
    where
        F: FnMut(WebEvent) + 'static,
    {
        event::listener::listen(self.element.clone().unchecked_into(), event_name, callback)
    }
}
