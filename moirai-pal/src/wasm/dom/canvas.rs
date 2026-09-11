//! Owned HTML5 canvas presentation through the browser PAL.

use super::{WebDocument, WebElement};
use std::io;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

pub use crate::canvas_validation::{CanvasSize, RgbaFrame};

/// A browser canvas with a retained two-dimensional rendering context.
pub struct WebCanvas {
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
}

impl WebCanvas {
    /// Wraps an existing DOM element as a canvas.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the element is not a canvas
    /// or the browser cannot provide a two-dimensional context.
    pub fn from_element(element: &WebElement) -> io::Result<Self> {
        let canvas = element
            .element
            .clone()
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "DOM element is not a canvas")
            })?;
        let context = canvas
            .get_context("2d")
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "canvas context lookup failed")
            })?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "canvas has no 2-D context")
            })?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "canvas context is not 2-D")
            })?;
        Ok(Self { canvas, context })
    }

    /// Returns the canvas element's stable identifier.
    #[must_use]
    pub fn id(&self) -> String {
        self.canvas.id()
    }

    /// Presents one validated RGBA8 frame at the canvas origin.
    ///
    /// The frame remains borrowed for the duration of this call. The browser
    /// owns any transfer needed by `ImageData` after the Web API boundary.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser rejects the
    /// canvas resize or image upload.
    pub fn present(&self, frame: RgbaFrame<'_>) -> io::Result<()> {
        let size = frame.size();
        self.canvas.set_width(size.width());
        self.canvas.set_height(size.height());
        let image = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(frame.pixels()),
            size.width(),
            size.height(),
        )
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "browser rejected RGBA image"))?;
        self.context.put_image_data(&image, 0.0, 0.0).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser rejected canvas upload",
            )
        })
    }
}

impl WebDocument {
    /// Resolves a canvas by identifier in this document.
    ///
    /// # Errors
    /// Propagates the missing-element, element-kind and context errors from
    /// [`WebCanvas::from_element`].
    pub fn canvas_by_id(&self, id: &str) -> io::Result<WebCanvas> {
        let element = self.get_element_by_id(id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "canvas element identifier is absent",
            )
        })?;
        WebCanvas::from_element(&element)
    }
}
