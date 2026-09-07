//! Safe, owned browser DOM handles for Atlas applications.

use std::io;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use web_sys::{
    Document, Element, Event, HtmlButtonElement, HtmlInputElement, HtmlSelectElement, Window,
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
