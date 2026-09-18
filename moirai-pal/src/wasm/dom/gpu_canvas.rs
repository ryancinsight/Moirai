//! Explicit browser WebGPU presentation through the browser PAL.

use super::{WebDocument, WebElement};
use js_sys::{Function, Object, Promise, Reflect};
use std::cell::Cell;
use std::io;
use wasm_bindgen::{Clamped, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{HtmlCanvasElement, ImageData};

use crate::canvas_validation::{CanvasSize, RgbaFrame};

const GPU_TEXTURE_USAGE_COPY_DST: f64 = 2.0;
const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT: f64 = 16.0;
const GPU_TEXTURE_USAGE_CANVAS: f64 =
    GPU_TEXTURE_USAGE_COPY_DST + GPU_TEXTURE_USAGE_RENDER_ATTACHMENT;

/// A browser canvas configured for explicit WebGPU frame presentation.
///
/// The surface owns the browser device, queue and swap-chain context. Frame
/// bytes remain borrowed from the caller and are copied directly into the
/// current canvas texture by `GPUQueue.copyExternalImageToTexture`.
pub struct WebGpuCanvas {
    canvas: HtmlCanvasElement,
    context: JsValue,
    queue: JsValue,
    device: JsValue,
    format: String,
    configured_size: Cell<Option<CanvasSize>>,
}

struct GpuState {
    device: JsValue,
    queue: JsValue,
    format: String,
}

impl WebGpuCanvas {
    /// Resolves a canvas element and asynchronously acquires a WebGPU device.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the element is not a
    /// canvas, [`io::ErrorKind::Unsupported`] when the browser exposes no
    /// WebGPU adapter, or a typed I/O error when device setup fails.
    pub async fn from_element(element: &WebElement) -> io::Result<Self> {
        let canvas = element
            .element
            .clone()
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "DOM element is not a canvas")
            })?;
        let context = canvas
            .get_context("webgpu")
            .map_err(|error| browser_error("get WebGPU canvas context", error))?
            .map(JsValue::from)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "browser has no WebGPU canvas context",
                )
            })?;

        let state = acquire_gpu_state().await?;
        Ok(Self {
            canvas,
            context,
            queue: state.queue,
            device: state.device,
            format: state.format,
            configured_size: Cell::new(None),
        })
    }

    /// Resolves a named canvas in the current document and acquires WebGPU.
    ///
    /// # Errors
    /// Propagates document lookup, adapter, and device setup failures from
    /// [`Self::from_element`].
    pub async fn from_current_document(id: &str) -> io::Result<Self> {
        let document = WebDocument::current()?;
        let element = document.get_element_by_id(id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "canvas element identifier is absent",
            )
        })?;
        Self::from_element(&element).await
    }

    /// Returns the resolved canvas identifier.
    #[must_use]
    pub fn id(&self) -> String {
        self.canvas.id()
    }

    /// Replaces a lost browser device and clears the configured extent.
    ///
    /// Recovery is explicit and never changes this surface to the two-dimensional
    /// presenter. The old device state remains in place when adapter or device
    /// setup fails, so a caller can surface the typed error and decide whether
    /// to retry or close the surface.
    ///
    /// # Errors
    /// Returns the same typed browser setup errors as [`Self::from_element`].
    pub async fn recreate(&mut self) -> io::Result<()> {
        let state = acquire_gpu_state().await?;
        self.device = state.device;
        self.queue = state.queue;
        self.format = state.format;
        self.configured_size.set(None);
        Ok(())
    }

    /// Presents one validated RGBA8 frame through the WebGPU canvas texture.
    ///
    /// The canvas is resized and reconfigured only when the frame extent
    /// changes. The source `ImageData` and destination texture are temporary
    /// browser objects; this surface retains no pixel bytes after submission.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] for an invalid frame or when a
    /// browser operation rejects the upload, and [`io::ErrorKind::Other`] for
    /// a lost or incomplete WebGPU surface.
    pub fn present(&self, frame: RgbaFrame<'_>) -> io::Result<()> {
        let size = frame.size();
        if self.configured_size.get() != Some(size) {
            self.canvas.set_width(size.width());
            self.canvas.set_height(size.height());
            self.configure()?;
            self.configured_size.set(Some(size));
        }

        let image = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(frame.pixels()),
            size.width(),
            size.height(),
        )
        .map_err(|error| browser_error("create WebGPU ImageData source", error))?;
        let texture = call_method(&self.context, "getCurrentTexture", &[])?;
        let source = Object::new();
        set_property(&source, "source", &image.into())?;
        let destination = Object::new();
        set_property(&destination, "texture", &texture)?;
        set_property(&destination, "colorSpace", &JsValue::from_str("srgb"))?;
        set_property(&destination, "premultipliedAlpha", &JsValue::FALSE)?;
        let copy_size = Object::new();
        set_property(
            &copy_size,
            "width",
            &JsValue::from_f64(f64::from(size.width())),
        )?;
        set_property(
            &copy_size,
            "height",
            &JsValue::from_f64(f64::from(size.height())),
        )?;
        set_property(&copy_size, "depthOrArrayLayers", &JsValue::from_f64(1.0))?;
        call_method(
            &self.queue,
            "copyExternalImageToTexture",
            &[source.into(), destination.into(), copy_size.into()],
        )?;
        Ok(())
    }

    fn configure(&self) -> io::Result<()> {
        let configuration = Object::new();
        set_property(&configuration, "device", &self.device)?;
        set_property(&configuration, "format", &JsValue::from_str(&self.format))?;
        set_property(
            &configuration,
            "usage",
            &JsValue::from_f64(GPU_TEXTURE_USAGE_CANVAS),
        )?;
        set_property(&configuration, "alphaMode", &JsValue::from_str("opaque"))?;
        call_method(&self.context, "configure", &[configuration.into()])?;
        Ok(())
    }
}

async fn acquire_gpu_state() -> io::Result<GpuState> {
    let window = web_sys::window().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "WebGPU setup requires a browser Window",
        )
    })?;
    let window_value: JsValue = window.into();
    let navigator = property(&window_value, "navigator")?;
    let gpu = property_or_unsupported(&navigator, "gpu", "browser exposes no WebGPU")?;
    let adapter_request = call_method(&gpu, "requestAdapter", &[])?;
    let adapter = await_promise(adapter_request, "request WebGPU adapter").await?;
    if adapter.is_null() || adapter.is_undefined() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "browser returned no WebGPU adapter",
        ));
    }
    let device_request = call_method(&adapter, "requestDevice", &[])?;
    let device = await_promise(device_request, "request WebGPU device").await?;
    let queue = property(&device, "queue")?;
    let format = call_method(&gpu, "getPreferredCanvasFormat", &[])?
        .as_string()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "WebGPU preferred canvas format is not a string",
            )
        })?;
    Ok(GpuState {
        device,
        queue,
        format,
    })
}

impl WebDocument {
    /// Resolves a named canvas and asynchronously acquires a WebGPU device.
    ///
    /// # Errors
    /// Propagates the missing-element and WebGPU setup errors from
    /// [`WebGpuCanvas::from_element`].
    pub async fn gpu_canvas_by_id(&self, id: &str) -> io::Result<WebGpuCanvas> {
        let element = self.get_element_by_id(id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "canvas element identifier is absent",
            )
        })?;
        WebGpuCanvas::from_element(&element).await
    }
}

fn property(receiver: &JsValue, name: &str) -> io::Result<JsValue> {
    Reflect::get(receiver, &JsValue::from_str(name))
        .map_err(|error| browser_error(name, error))
        .and_then(|value| {
            if value.is_undefined() || value.is_null() {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("WebGPU property {name:?} is absent"),
                ))
            } else {
                Ok(value)
            }
        })
}

fn property_or_unsupported(receiver: &JsValue, name: &str, message: &str) -> io::Result<JsValue> {
    Reflect::get(receiver, &JsValue::from_str(name))
        .map_err(|error| browser_error(name, error))
        .and_then(|value| {
            if value.is_undefined() || value.is_null() {
                Err(io::Error::new(io::ErrorKind::Unsupported, message))
            } else {
                Ok(value)
            }
        })
}

fn call_method(receiver: &JsValue, name: &str, args: &[JsValue]) -> io::Result<JsValue> {
    let function = property(receiver, name)?
        .dyn_into::<Function>()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("WebGPU property {name:?} is not callable"),
            )
        })?;
    let arguments = js_sys::Array::new();
    for argument in args {
        arguments.push(argument);
    }
    function
        .apply(receiver, &arguments)
        .map_err(|error| browser_error(name, error))
}

fn set_property(object: &Object, name: &str, value: &JsValue) -> io::Result<()> {
    Reflect::set(object, &JsValue::from_str(name), value)
        .map(|_| ())
        .map_err(|error| browser_error(name, error))
}

async fn await_promise(value: JsValue, operation: &'static str) -> io::Result<JsValue> {
    let promise = value.dyn_into::<Promise>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{operation} returned a non-Promise value"),
        )
    })?;
    JsFuture::from(promise)
        .await
        .map_err(|error| browser_error(operation, error))
}

fn browser_error(operation: &str, error: JsValue) -> io::Error {
    let detail = error
        .as_string()
        .unwrap_or_else(|| "browser returned a JavaScript exception".to_owned());
    io::Error::other(format!("{operation}: {detail}"))
}
