//! Owned events, pointer snapshots and listener lifetimes.
use super::WebElement;
use std::io;
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{Element, Event, MouseEvent, PointerEvent, WheelEvent};

/// A browser event delivered to an owned listener.
pub struct WebEvent {
    pub(super) event: Event,
}

impl WebEvent {
    /// Returns the browser's trust snapshot for this event.
    ///
    /// `true` means the browser marked the event as trusted. This is a
    /// browser provenance signal, not proof of a physical user or of an
    /// authenticated automation session; consumers apply their own input
    /// policy.
    #[must_use]
    pub fn is_trusted(&self) -> bool {
        self.event.is_trusted()
    }
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
    pub(super) ctrl: bool,
    pub(super) shift: bool,
    pub(super) alt: bool,
    pub(super) meta: bool,
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerMetadata {
    pointer_id: i32,
    pointer_type: PointerType,
    client_x: f64,
    client_y: f64,
    offset_x: i32,
    offset_y: i32,
    button: i16,
    buttons: u16,
    modifiers: PointerModifiers,
    primary: bool,
    trusted: bool,
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
    pub const fn client_x(self) -> f64 {
        self.client_x
    }

    /// Returns the viewport-relative vertical coordinate in CSS pixels.
    #[must_use]
    pub const fn client_y(self) -> f64 {
        self.client_y
    }

    /// Returns the target-relative horizontal coordinate in CSS pixels.
    #[must_use]
    pub const fn offset_x(self) -> i32 {
        self.offset_x
    }

    /// Returns the target-relative vertical coordinate in CSS pixels.
    #[must_use]
    pub const fn offset_y(self) -> i32 {
        self.offset_y
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

    /// Returns the browser trust snapshot captured with this event.
    #[must_use]
    pub const fn is_trusted(self) -> bool {
        self.trusted
    }
}

/// Input metadata captured from one browser wheel event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelMetadata {
    delta_x: f64,
    delta_y: f64,
    delta_z: f64,
    delta_mode: WheelDeltaMode,
    client_x: f64,
    client_y: f64,
    offset_x: i32,
    offset_y: i32,
    modifiers: PointerModifiers,
    trusted: bool,
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
    pub const fn client_x(self) -> f64 {
        self.client_x
    }

    /// Returns the viewport-relative vertical coordinate in CSS pixels.
    #[must_use]
    pub const fn client_y(self) -> f64 {
        self.client_y
    }

    /// Returns the target-relative horizontal coordinate in CSS pixels.
    #[must_use]
    pub const fn offset_x(self) -> i32 {
        self.offset_x
    }

    /// Returns the target-relative vertical coordinate in CSS pixels.
    #[must_use]
    pub const fn offset_y(self) -> i32 {
        self.offset_y
    }

    /// Returns the modifier-key snapshot.
    #[must_use]
    pub const fn modifiers(self) -> PointerModifiers {
        self.modifiers
    }

    /// Returns the browser trust snapshot captured with this event.
    #[must_use]
    pub const fn is_trusted(self) -> bool {
        self.trusted
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
    /// The snapshot includes the pointer device, viewport and target-relative
    /// coordinates, button state, modifier keys, primary-pointer marker and
    /// browser trust provenance.
    /// Events that are not [`PointerEvent`] values return [`None`].
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
            client_x: event_coordinate(mouse, "clientX")?,
            client_y: event_coordinate(mouse, "clientY")?,
            offset_x: MouseEvent::offset_x(mouse),
            offset_y: MouseEvent::offset_y(mouse),
            button: MouseEvent::button(mouse),
            buttons: MouseEvent::buttons(mouse),
            modifiers: modifier_state(mouse),
            primary: PointerEvent::is_primary(pointer),
            trusted: self.is_trusted(),
        })
    }

    /// Reads wheel metadata from this event.
    ///
    /// The snapshot includes three browser deltas, their unit, viewport and
    /// target-relative coordinates, modifier keys and browser trust
    /// provenance. Events that are not [`WheelEvent`] values return [`None`].
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
            client_x: event_coordinate(mouse, "clientX")?,
            client_y: event_coordinate(mouse, "clientY")?,
            offset_x: MouseEvent::offset_x(mouse),
            offset_y: MouseEvent::offset_y(mouse),
            modifiers: modifier_state(mouse),
            trusted: self.is_trusted(),
        })
    }

    /// Stops the browser's default action for this event.
    pub fn prevent_default(&self) {
        self.event.prevent_default();
    }
}

fn event_coordinate(event: &MouseEvent, property: &str) -> Option<f64> {
    let value = js_sys::Reflect::get(event.as_ref(), &JsValue::from_str(property)).ok()?;
    let coordinate = value.as_f64()?;
    coordinate.is_finite().then_some(coordinate)
}

/// An event listener with explicit browser callback teardown.
pub struct WebEventListener {
    target: web_sys::EventTarget,
    event_name: String,
    closure: Closure<dyn FnMut(Event)>,
}

pub(super) fn listen<F>(
    target: web_sys::EventTarget,
    event_name: &str,
    mut callback: F,
) -> io::Result<WebEventListener>
where
    F: FnMut(WebEvent) + 'static,
{
    // Condition 1: the Web API callback boundary erases the closure type.
    let closure = Closure::wrap(
        Box::new(move |event: Event| callback(WebEvent { event })) as Box<dyn FnMut(Event)>
    );
    target
        .add_event_listener_with_callback(event_name, closure.as_ref().unchecked_ref())
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser rejected event listener registration",
            )
        })?;
    Ok(WebEventListener {
        target,
        event_name: event_name.to_owned(),
        closure,
    })
}

impl Drop for WebEventListener {
    fn drop(&mut self) {
        let _ = self.target.remove_event_listener_with_callback(
            &self.event_name,
            self.closure.as_ref().unchecked_ref(),
        );
    }
}
