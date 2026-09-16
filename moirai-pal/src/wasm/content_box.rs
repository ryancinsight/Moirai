//! Geometry for mapping viewport input into an element's local content box.

use std::io;

/// A viewport point expressed in an element's untransformed content box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentBoxPoint {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl ContentBoxPoint {
    /// Returns the horizontal coordinate from the content edge in CSS pixels.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns the vertical coordinate from the content edge in CSS pixels.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Returns the measured untransformed content width in CSS pixels.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    /// Returns the measured untransformed content height in CSS pixels.
    #[must_use]
    pub const fn height(self) -> f64 {
        self.height
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Affine2d {
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) c: f64,
    pub(crate) d: f64,
}

impl Affine2d {
    pub(crate) const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
    };

    pub(crate) fn then(self, outer: Self) -> Self {
        Self {
            a: outer.a.mul_add(self.a, outer.c * self.b),
            b: outer.b.mul_add(self.a, outer.d * self.b),
            c: outer.a.mul_add(self.c, outer.c * self.d),
            d: outer.b.mul_add(self.c, outer.d * self.d),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ContentBoxGeometry {
    content_width: f64,
    content_height: f64,
    content_left: f64,
    content_top: f64,
    transform: Affine2d,
    viewport_origin_x: f64,
    viewport_origin_y: f64,
}

impl ContentBoxGeometry {
    #[expect(
        clippy::too_many_arguments,
        reason = "the browser measurement has independent box, transform and origin observations"
    )]
    pub(crate) fn from_measurement(
        content_width: f64,
        content_height: f64,
        content_left: f64,
        content_top: f64,
        border_width: f64,
        border_height: f64,
        transform: Affine2d,
        rect_left: f64,
        rect_top: f64,
    ) -> io::Result<Self> {
        let scalars = [
            content_width,
            content_height,
            content_left,
            content_top,
            border_width,
            border_height,
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            rect_left,
            rect_top,
        ];
        if scalars.iter().any(|value| !value.is_finite())
            || content_width <= 0.0
            || content_height <= 0.0
            || border_width <= 0.0
            || border_height <= 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "browser content-box geometry is not finite and positive",
            ));
        }
        let determinant = transform
            .a
            .mul_add(transform.d, -(transform.b * transform.c));
        // The sum bounds first-order roundoff in the two products and their
        // subtraction without rejecting uniformly small, well-conditioned
        // transforms. Eight epsilons covers the two products, subtraction and
        // the browser-to-Rust coefficient transfer.
        let roundoff_bound = f64::EPSILON
            * 8.0
            * ((transform.a * transform.d).abs() + (transform.b * transform.c).abs());
        if !determinant.is_finite() || determinant.abs() <= roundoff_bound {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "browser content-box transform is singular",
            ));
        }

        let x_corners = [
            0.0,
            transform.a * border_width,
            transform.c * border_height,
            transform
                .a
                .mul_add(border_width, transform.c * border_height),
        ];
        let y_corners = [
            0.0,
            transform.b * border_width,
            transform.d * border_height,
            transform
                .b
                .mul_add(border_width, transform.d * border_height),
        ];
        let min_x = x_corners.into_iter().fold(f64::INFINITY, f64::min);
        let min_y = y_corners.into_iter().fold(f64::INFINITY, f64::min);

        Ok(Self {
            content_width,
            content_height,
            content_left,
            content_top,
            transform,
            viewport_origin_x: rect_left - min_x,
            viewport_origin_y: rect_top - min_y,
        })
    }

    pub(crate) fn point(self, client_x: f64, client_y: f64) -> io::Result<ContentBoxPoint> {
        if !client_x.is_finite() || !client_y.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser client point is not finite",
            ));
        }
        let x = client_x - self.viewport_origin_x;
        let y = client_y - self.viewport_origin_y;
        let determinant = self
            .transform
            .a
            .mul_add(self.transform.d, -(self.transform.b * self.transform.c));
        let border_x = self.transform.d.mul_add(x, -(self.transform.c * y)) / determinant;
        let border_y = (-self.transform.b).mul_add(x, self.transform.a * y) / determinant;
        Ok(ContentBoxPoint {
            x: border_x - self.content_left,
            y: border_y - self.content_top,
            width: self.content_width,
            height: self.content_height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Affine2d, ContentBoxGeometry};
    use std::io;

    #[test]
    fn border_and_padding_are_excluded_from_identity_coordinates() {
        let geometry = ContentBoxGeometry::from_measurement(
            320.5,
            180.25,
            13.5,
            9.25,
            348.0,
            201.0,
            Affine2d::IDENTITY,
            40.0,
            60.0,
        )
        .expect("identity box is valid");
        let point = geometry
            .point(53.5 + 80.125, 69.25 + 45.5)
            .expect("client point is finite");
        assert_eq!(point.x(), 80.125);
        assert_eq!(point.y(), 45.5);
        assert_eq!(point.width(), 320.5);
        assert_eq!(point.height(), 180.25);
    }

    #[test]
    fn nested_affine_transform_maps_back_to_fractional_local_point() {
        let child = Affine2d {
            a: 1.0,
            b: 0.25,
            c: -0.5,
            d: 1.0,
        };
        let ancestor = Affine2d {
            a: 1.5,
            b: 0.0,
            c: 0.0,
            d: 0.75,
        };
        let transform = child.then(ancestor);
        let border_width = 220.0;
        let border_height = 140.0;
        let geometry = ContentBoxGeometry::from_measurement(
            200.0,
            120.0,
            12.0,
            8.0,
            border_width,
            border_height,
            transform,
            75.0,
            32.0,
        )
        .expect("affine box is valid");

        let border_x: f64 = 12.0 + 47.75;
        let border_y: f64 = 8.0 + 63.125;
        let min_x = (transform.c * border_height).min(0.0);
        let client_x = 75.0 - min_x + transform.a * border_x + transform.c * border_y;
        let client_y = 32.0 + transform.b * border_x + transform.d * border_y;
        let point = geometry
            .point(client_x, client_y)
            .expect("transformed point is finite");
        assert!((point.x() - 47.75).abs() <= f64::EPSILON * 64.0);
        assert!((point.y() - 63.125).abs() <= f64::EPSILON * 64.0);
        assert_eq!(point.width(), 200.0);
        assert_eq!(point.height(), 120.0);
    }

    #[test]
    fn singular_geometry_is_rejected_without_rejecting_small_scales() {
        let singular = Affine2d {
            a: 1.0,
            b: 2.0,
            c: 0.5,
            d: 1.0,
        };
        let error = ContentBoxGeometry::from_measurement(
            10.0, 10.0, 0.0, 0.0, 10.0, 10.0, singular, 0.0, 0.0,
        )
        .expect_err("singular transform must be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "browser content-box transform is singular"
        );
        let small = Affine2d {
            a: 1.0e-9,
            b: 0.0,
            c: 0.0,
            d: 1.0e-9,
        };
        let geometry =
            ContentBoxGeometry::from_measurement(7.0, 8.0, 1.25, 0.75, 10.0, 10.0, small, 0.0, 0.0)
                .expect("small invertible scale is valid");
        let expected_x = 2.5;
        let expected_y = 3.125;
        let point = geometry
            .point(small.a * (1.25 + expected_x), small.d * (0.75 + expected_y))
            .expect("scaled client point is finite");
        // Each recovered coordinate uses one multiply, a 2x2 determinant and
        // one division. Eight roundings bound this diagonal case.
        let coordinate_bound = 8.0 * f64::EPSILON * expected_y;
        assert!((point.x() - expected_x).abs() <= coordinate_bound);
        assert!((point.y() - expected_y).abs() <= coordinate_bound);
        assert_eq!(point.width(), 7.0);
        assert_eq!(point.height(), 8.0);
    }
}
