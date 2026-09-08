//! Safe, owned browser DOM handles for Atlas applications.

mod file_drop;

pub use self::file_drop::{DropMetadata, DroppedFile};

use std::io;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use web_sys::{
    Document, Element, Event, HtmlButtonElement, HtmlDialogElement, HtmlElement, HtmlInputElement,
    HtmlSelectElement, MouseEvent, PointerEvent, WheelEvent, Window,
};

/// A browser document obtained from the current window.
#[derive(Clone)]
pub struct WebDocument {
    document: Document,
}

impl WebDocument {
    /// Obtains the document for the current browser window.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when no browser window or
    /// document is available.
    pub fn current() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "DOM access requires a browser Window",
            )
        })?;
        Self::from_window(window)
    }

    /// Wraps a browser window's document.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when the window has no document.
    pub fn from_window(window: Window) -> io::Result<Self> {
        window
            .document()
            .map(|document| Self { document })
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::Unsupported, "Browser Window has no Document")
            })
    }

    /// Returns the document body.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::NotFound`] before the document body exists.
    pub fn body(&self) -> io::Result<WebElement> {
        self.document
            .body()
            .map(|body| WebElement {
                element: body.unchecked_into(),
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Document body is absent"))
    }

    /// Finds an element by its stable identifier.
    #[must_use]
    pub fn get_element_by_id(&self, id: &str) -> Option<WebElement> {
        self.document
            .get_element_by_id(id)
            .map(|element| WebElement { element })
    }

    /// Creates an element with the supplied HTML tag name.
    ///
    /// # Errors
    /// Returns the browser's DOM validation error as [`io::ErrorKind::InvalidInput`].
    pub fn create_element(&self, tag: &str) -> io::Result<WebElement> {
        self.document
            .create_element(tag)
            .map(|element| WebElement { element })
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid DOM element name"))
    }
}

/// An owned browser element handle.
#[derive(Clone)]
pub struct WebElement {
    element: Element,
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

    /// Replaces the value of a browser input element.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the element is not an input.
    pub fn set_value(&self, value: &str) -> io::Result<()> {
        let input = self.element.dyn_ref::<HtmlInputElement>().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "DOM element is not an input")
        })?;
        input.set_value(value);
        Ok(())
    }

    /// Registers a callback whose lifetime is tied to the returned listener.
    ///
    /// Dropping the listener removes the callback from the element before the
    /// closure is released, so cancellation cannot leave a JavaScript root.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the event name is invalid
    /// or the browser refuses the listener.
    pub fn add_event_listener<F>(
        &self,
        event_name: &str,
        callback: F,
    ) -> io::Result<WebEventListener>
    where
        F: FnMut(WebEvent) + 'static,
    {
        let mut callback = callback;
        let closure = Closure::wrap(Box::new(move |event: Event| {
            callback(WebEvent { event });
        }) as Box<dyn FnMut(Event)>);
        self.element
            .add_event_listener_with_callback(event_name, closure.as_ref().unchecked_ref())
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Browser rejected event listener",
                )
            })?;
        Ok(WebEventListener {
            element: self.element.clone(),
            event_name: event_name.to_owned(),
            closure,
        })
    }
}

/// A browser event delivered to an owned listener.
pub struct WebEvent {
    event: Event,
}

/// The browser pointer device that produced an event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PointerType {
    /// A mouse or mouse-like pointing device.
    Mouse,
    /// A pen or stylus device.
    Pen,
    /// A direct-touch device.
    Touch,
    /// A browser-defined pointer type not covered by the known variants.
    Other,
}

impl PointerType {
    /// Returns the stable label used by the browser-facing documentation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mouse => "mouse",
            Self::Pen => "pen",
            Self::Touch => "touch",
            Self::Other => "other",
        }
    }
}

/// The unit used for browser wheel deltas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WheelDeltaMode {
    /// Deltas are expressed in CSS pixels.
    Pixel,
    /// Deltas are expressed in lines of content.
    Line,
    /// Deltas are expressed in pages of content.
    Page,
    /// A browser-defined delta unit.
    Other,
}

impl WheelDeltaMode {
    /// Returns the stable label used by the browser-facing documentation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pixel => "pixel",
            Self::Line => "line",
            Self::Page => "page",
            Self::Other => "other",
        }
    }
}

/// Modifier-key state captured with one browser input event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointerModifiers {
    ctrl: bool,
    shift: bool,
    alt: bool,
    meta: bool,
}

impl PointerModifiers {
    /// Returns whether Control was held for the event.
    #[must_use]
    pub const fn ctrl(self) -> bool {
        self.ctrl
    }

    /// Returns whether Shift was held for the event.
    #[must_use]
    pub const fn shift(self) -> bool {
        self.shift
    }

    /// Returns whether Alt was held for the event.
    #[must_use]
    pub const fn alt(self) -> bool {
        self.alt
    }

    /// Returns whether Meta was held for the event.
    #[must_use]
    pub const fn meta(self) -> bool {
        self.meta
    }
}

/// Input metadata captured from one browser pointer event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointerMetadata {
    pointer_id: i32,
    pointer_type: PointerType,
    client_x: i32,
    client_y: i32,
    button: i16,
    buttons: u16,
    modifiers: PointerModifiers,
    primary: bool,
}

impl PointerMetadata {
    /// Returns the browser-assigned pointer identifier.
    #[must_use]
    pub const fn pointer_id(self) -> i32 {
        self.pointer_id
    }

    /// Returns the normalized pointer device type.
    #[must_use]
    pub const fn pointer_type(self) -> PointerType {
        self.pointer_type
    }

    /// Returns the viewport-relative horizontal coordinate in CSS pixels.
    #[must_use]
    pub const fn client_x(self) -> i32 {
        self.client_x
    }

    /// Returns the viewport-relative vertical coordinate in CSS pixels.
    #[must_use]
    pub const fn client_y(self) -> i32 {
        self.client_y
    }

    /// Returns the button changed by the event (`-1` when the browser has no button).
    #[must_use]
    pub const fn button(self) -> i16 {
        self.button
    }

    /// Returns the bitmask of buttons currently held down.
    #[must_use]
    pub const fn buttons(self) -> u16 {
        self.buttons
    }

    /// Returns the modifier-key snapshot.
    #[must_use]
    pub const fn modifiers(self) -> PointerModifiers {
        self.modifiers
    }

    /// Returns whether this is the primary pointer for its device.
    #[must_use]
    pub const fn is_primary(self) -> bool {
        self.primary
    }
}

/// Input metadata captured from one browser wheel event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelMetadata {
    delta_x: f64,
    delta_y: f64,
    delta_z: f64,
    delta_mode: WheelDeltaMode,
    client_x: i32,
    client_y: i32,
    modifiers: PointerModifiers,
}

impl WheelMetadata {
    /// Returns the horizontal wheel delta in [`Self::delta_mode`] units.
    #[must_use]
    pub const fn delta_x(self) -> f64 {
        self.delta_x
    }

    /// Returns the vertical wheel delta in [`Self::delta_mode`] units.
    #[must_use]
    pub const fn delta_y(self) -> f64 {
        self.delta_y
    }

    /// Returns the depth wheel delta in [`Self::delta_mode`] units.
    #[must_use]
    pub const fn delta_z(self) -> f64 {
        self.delta_z
    }

    /// Returns the browser unit used by the deltas.
    #[must_use]
    pub const fn delta_mode(self) -> WheelDeltaMode {
        self.delta_mode
    }

    /// Returns the viewport-relative horizontal coordinate in CSS pixels.
    #[must_use]
    pub const fn client_x(self) -> i32 {
        self.client_x
    }

    /// Returns the viewport-relative vertical coordinate in CSS pixels.
    #[must_use]
    pub const fn client_y(self) -> i32 {
        self.client_y
    }

    /// Returns the modifier-key snapshot.
    #[must_use]
    pub const fn modifiers(self) -> PointerModifiers {
        self.modifiers
    }
}

fn modifier_state(event: &MouseEvent) -> PointerModifiers {
    PointerModifiers {
        ctrl: MouseEvent::ctrl_key(event),
        shift: MouseEvent::shift_key(event),
        alt: MouseEvent::alt_key(event),
        meta: MouseEvent::meta_key(event),
    }
}

impl WebEvent {
    /// Returns the event target when it is a DOM element.
    #[must_use]
    pub fn target(&self) -> Option<WebElement> {
        self.event
            .target()
            .and_then(|target| target.dyn_into::<Element>().ok())
            .map(|element| WebElement { element })
    }

    /// Reads an input or select value from the event target.
    #[must_use]
    pub fn value(&self) -> Option<String> {
        self.target().and_then(|target| target.value())
    }

    /// Returns the identifier carried by a pointer event.
    #[must_use]
    pub fn pointer_id(&self) -> Option<i32> {
        self.event
            .dyn_ref::<PointerEvent>()
            .map(PointerEvent::pointer_id)
    }

    /// Reads pointer metadata from this event.
    ///
    /// The snapshot includes the pointer device, viewport coordinates, button
    /// state, modifier keys and primary-pointer marker. Events that are not
    /// [`PointerEvent`] values return [`None`].
    #[must_use]
    pub fn pointer_metadata(&self) -> Option<PointerMetadata> {
        let pointer = self.event.dyn_ref::<PointerEvent>()?;
        let mouse = self.event.dyn_ref::<MouseEvent>()?;
        let pointer_type = match PointerEvent::pointer_type(pointer).as_str() {
            "mouse" => PointerType::Mouse,
            "pen" => PointerType::Pen,
            "touch" => PointerType::Touch,
            _ => PointerType::Other,
        };
        Some(PointerMetadata {
            pointer_id: PointerEvent::pointer_id(pointer),
            pointer_type,
            client_x: MouseEvent::client_x(mouse),
            client_y: MouseEvent::client_y(mouse),
            button: MouseEvent::button(mouse),
            buttons: MouseEvent::buttons(mouse),
            modifiers: modifier_state(mouse),
            primary: PointerEvent::is_primary(pointer),
        })
    }

    /// Reads wheel metadata from this event.
    ///
    /// The snapshot includes three browser deltas, their unit, viewport
    /// coordinates and modifier keys. Events that are not [`WheelEvent`]
    /// values return [`None`].
    #[must_use]
    pub fn wheel_metadata(&self) -> Option<WheelMetadata> {
        let wheel = self.event.dyn_ref::<WheelEvent>()?;
        let mouse = self.event.dyn_ref::<MouseEvent>()?;
        let delta_mode = match WheelEvent::delta_mode(wheel) {
            0 => WheelDeltaMode::Pixel,
            1 => WheelDeltaMode::Line,
            2 => WheelDeltaMode::Page,
            _ => WheelDeltaMode::Other,
        };
        Some(WheelMetadata {
            delta_x: WheelEvent::delta_x(wheel),
            delta_y: WheelEvent::delta_y(wheel),
            delta_z: WheelEvent::delta_z(wheel),
            delta_mode,
            client_x: MouseEvent::client_x(mouse),
            client_y: MouseEvent::client_y(mouse),
            modifiers: modifier_state(mouse),
        })
    }

    /// Stops the browser's default action for this event.
    pub fn prevent_default(&self) {
        self.event.prevent_default();
    }
}

/// An event listener with explicit browser callback teardown.
pub struct WebEventListener {
    element: Element,
    event_name: String,
    closure: Closure<dyn FnMut(Event)>,
}

impl Drop for WebEventListener {
    fn drop(&mut self) {
        let _ = self.element.remove_event_listener_with_callback(
            &self.event_name,
            self.closure.as_ref().unchecked_ref(),
        );
    }
}
