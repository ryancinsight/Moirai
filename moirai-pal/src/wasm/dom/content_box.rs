//! Resolved CSS measurement for local content-box input coordinates.

use crate::content_box::{Affine2d, ContentBoxGeometry, ContentBoxPoint};
use std::io;
use wasm_bindgen::JsCast;
use web_sys::{CssStyleDeclaration, DomMatrixReadOnly, Element, ShadowRoot, Window};

#[derive(Clone, Copy)]
struct BoxDimensions {
    content_width: f64,
    content_height: f64,
    content_left: f64,
    content_top: f64,
    border_width: f64,
    border_height: f64,
}

pub(super) fn measure_point(
    element: &Element,
    client_x: f64,
    client_y: f64,
) -> io::Result<ContentBoxPoint> {
    let window = browser_window()?;
    let dimensions = content_box_dimensions(&computed_style(&window, element)?)?;
    let mut transform = Affine2d::IDENTITY;
    let mut ancestor = Some(element.clone());
    while let Some(current) = ancestor {
        let style = computed_style(&window, &current)?;
        reject_unmodeled_transform_properties(&style)?;
        if css_transform_applies(&current, &style)? {
            transform = transform.then(affine_transform(&style)?);
        }
        ancestor = composed_parent(&current);
    }

    let bounds = element.get_bounding_client_rect();
    ContentBoxGeometry::from_measurement(
        dimensions.content_width,
        dimensions.content_height,
        dimensions.content_left,
        dimensions.content_top,
        dimensions.border_width,
        dimensions.border_height,
        transform,
        bounds.left(),
        bounds.top(),
    )?
    .point(client_x, client_y)
}

fn browser_window() -> io::Result<Window> {
    web_sys::window().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "content-box measurement requires a browser Window",
        )
    })
}

fn computed_style(window: &Window, element: &Element) -> io::Result<CssStyleDeclaration> {
    window
        .get_computed_style(element)
        .map_err(|_| io::Error::other("browser rejected resolved CSS measurement"))?
        .ok_or_else(|| io::Error::other("browser returned no resolved CSS measurement"))
}

fn content_box_dimensions(style: &CssStyleDeclaration) -> io::Result<BoxDimensions> {
    let padding_left = css_pixels(style, "padding-left")?;
    let padding_right = css_pixels(style, "padding-right")?;
    let padding_top = css_pixels(style, "padding-top")?;
    let padding_bottom = css_pixels(style, "padding-bottom")?;
    let border_left = css_pixels(style, "border-left-width")?;
    let border_right = css_pixels(style, "border-right-width")?;
    let border_top = css_pixels(style, "border-top-width")?;
    let border_bottom = css_pixels(style, "border-bottom-width")?;
    let horizontal_insets = padding_left + padding_right + border_left + border_right;
    let vertical_insets = padding_top + padding_bottom + border_top + border_bottom;
    let resolved_width = css_pixels(style, "width")?;
    let resolved_height = css_pixels(style, "height")?;
    let border_box = css_property(style, "box-sizing")? == "border-box";
    let content_width = if border_box {
        resolved_width - horizontal_insets
    } else {
        resolved_width
    };
    let content_height = if border_box {
        resolved_height - vertical_insets
    } else {
        resolved_height
    };
    let values = [
        content_width,
        content_height,
        horizontal_insets,
        vertical_insets,
    ];
    if values.iter().any(|value| !value.is_finite())
        || content_width <= 0.0
        || content_height <= 0.0
        || horizontal_insets < 0.0
        || vertical_insets < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "resolved CSS content box is not finite and positive",
        ));
    }
    Ok(BoxDimensions {
        content_width,
        content_height,
        content_left: border_left + padding_left,
        content_top: border_top + padding_top,
        border_width: content_width + horizontal_insets,
        border_height: content_height + vertical_insets,
    })
}

fn affine_transform(style: &CssStyleDeclaration) -> io::Result<Affine2d> {
    let value = css_property(style, "transform")?;
    let transform = if value == "none" {
        Affine2d::IDENTITY
    } else {
        let matrix = DomMatrixReadOnly::new_with_str(&value).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "browser returned an invalid computed CSS transform",
            )
        })?;
        if !matrix.is_2d() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "three-dimensional CSS transforms cannot map canvas input",
            ));
        }
        Affine2d {
            a: matrix.a(),
            b: matrix.b(),
            c: matrix.c(),
            d: matrix.d(),
        }
    };
    // CSS Transforms Level 2 applies individual translate, rotate and scale
    // before the `transform` list. Translation changes only the viewport
    // origin, which the measured border rectangle supplies independently.
    let scale = individual_scale(style)?;
    let rotation = individual_rotation(style)?;
    Ok(transform.then(scale).then(rotation))
}

fn css_transform_applies(element: &Element, style: &CssStyleDeclaration) -> io::Result<bool> {
    // CSS Transforms excludes non-replaced inline boxes and the table column
    // boxes from transformable elements. `display: contents` generates no box.
    let display = css_property(style, "display")?;
    if matches!(
        display.as_str(),
        "contents" | "table-column" | "table-column-group"
    ) {
        return Ok(false);
    }
    if display != "inline" {
        return Ok(true);
    }
    Ok(matches!(
        element.tag_name().as_str(),
        "CANVAS" | "EMBED" | "IFRAME" | "IMG" | "INPUT" | "OBJECT" | "VIDEO"
    ))
}

fn composed_parent(element: &Element) -> Option<Element> {
    if let Some(slot) = element.assigned_slot() {
        return Some(slot.unchecked_into());
    }
    if let Some(parent) = element.parent_element() {
        return Some(parent);
    }
    element
        .get_root_node()
        .dyn_into::<ShadowRoot>()
        .ok()
        .map(|root| root.host())
}

fn reject_unmodeled_transform_properties(style: &CssStyleDeclaration) -> io::Result<()> {
    if css_property(style, "offset-path")? != "none" {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "CSS motion paths cannot map canvas input",
        ));
    }
    let perspective = css_property(style, "perspective")?;
    if perspective != "none" {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "perspective CSS transforms cannot map canvas input",
        ));
    }
    let zoom = css_property(style, "zoom")?;
    if !matches!(zoom.as_str(), "" | "1" | "normal") {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "CSS zoom cannot map canvas input",
        ));
    }
    Ok(())
}

fn individual_scale(style: &CssStyleDeclaration) -> io::Result<Affine2d> {
    let value = css_property(style, "scale")?;
    if value == "none" {
        return Ok(Affine2d::IDENTITY);
    }
    let values = value
        .split_ascii_whitespace()
        .map(css_scale_factor)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid_individual_transform())?;
    let (scale_x, scale_y) = match values.as_slice() {
        [scale] => (*scale, *scale),
        [scale_x, scale_y] | [scale_x, scale_y, _] => (*scale_x, *scale_y),
        _ => return Err(invalid_individual_transform()),
    };
    if !scale_x.is_finite() || !scale_y.is_finite() {
        return Err(invalid_individual_transform());
    }
    Ok(Affine2d {
        a: scale_x,
        b: 0.0,
        c: 0.0,
        d: scale_y,
    })
}

fn css_scale_factor(value: &str) -> Result<f64, std::num::ParseFloatError> {
    if let Some(percentage) = value.strip_suffix('%') {
        return percentage.parse::<f64>().map(|value| value / 100.0);
    }
    value.parse()
}

fn individual_rotation(style: &CssStyleDeclaration) -> io::Result<Affine2d> {
    let value = css_property(style, "rotate")?;
    if value == "none" {
        return Ok(Affine2d::IDENTITY);
    }
    let angle = value.strip_prefix("z ").unwrap_or(&value);
    if angle.split_ascii_whitespace().count() != 1 {
        return Err(invalid_individual_transform());
    }
    let radians = css_angle_radians(angle)?;
    let (sin, cos) = radians.sin_cos();
    Ok(Affine2d {
        a: cos,
        b: sin,
        c: -sin,
        d: cos,
    })
}

fn css_angle_radians(value: &str) -> io::Result<f64> {
    let (number, factor) = if let Some(number) = value.strip_suffix("deg") {
        (number, std::f64::consts::PI / 180.0)
    } else if let Some(number) = value.strip_suffix("grad") {
        (number, std::f64::consts::PI / 200.0)
    } else if let Some(number) = value.strip_suffix("rad") {
        (number, 1.0)
    } else if let Some(number) = value.strip_suffix("turn") {
        (number, std::f64::consts::TAU)
    } else {
        return Err(invalid_individual_transform());
    };
    let angle = number
        .parse::<f64>()
        .map_err(|_| invalid_individual_transform())?
        * factor;
    if !angle.is_finite() {
        return Err(invalid_individual_transform());
    }
    Ok(angle)
}

fn invalid_individual_transform() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "three-dimensional or invalid individual CSS transforms cannot map canvas input",
    )
}

fn css_pixels(style: &CssStyleDeclaration, property: &str) -> io::Result<f64> {
    let value = css_property(style, property)?;
    let pixels = value.strip_suffix("px").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "browser returned a non-pixel resolved CSS length",
        )
    })?;
    pixels.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "browser returned an invalid resolved CSS length",
        )
    })
}

fn css_property(style: &CssStyleDeclaration, property: &str) -> io::Result<String> {
    style
        .get_property_value(property)
        .map_err(|_| io::Error::other("browser rejected resolved CSS property access"))
}
