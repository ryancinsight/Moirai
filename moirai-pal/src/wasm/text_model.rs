//! Value types for browser text selections.

use std::io;

/// The direction of a browser text selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TextSelectionDirection {
    /// The selection anchor precedes its focus.
    Forward,
    /// The selection anchor follows its focus.
    Backward,
    /// The selection has no direction, usually because it is a caret.
    None,
    /// A browser-defined direction not covered by the known variants.
    Other,
}

impl TextSelectionDirection {
    /// Returns the browser spelling for this direction.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::None => "none",
            Self::Other => "other",
        }
    }

    pub(crate) fn from_browser(value: Option<&str>) -> Self {
        match value {
            Some("forward") => Self::Forward,
            Some("backward") => Self::Backward,
            Some("none") | None => Self::None,
            Some(_) => Self::Other,
        }
    }
}

/// A browser text selection expressed in UTF-16 code-unit offsets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextSelection {
    start: u32,
    end: u32,
    direction: TextSelectionDirection,
}

impl TextSelection {
    /// Creates a selection after checking its ordering invariant.
    ///
    /// # Errors
    /// Returns io::ErrorKind::InvalidInput when start follows end.
    pub fn new(start: u32, end: u32, direction: TextSelectionDirection) -> io::Result<Self> {
        if start > end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Text selection start follows end",
            ));
        }
        Ok(Self {
            start,
            end,
            direction,
        })
    }

    /// Returns the UTF-16 start offset.
    #[must_use]
    pub const fn start(self) -> u32 {
        self.start
    }

    /// Returns the UTF-16 end offset.
    #[must_use]
    pub const fn end(self) -> u32 {
        self.end
    }

    /// Returns the selection direction.
    #[must_use]
    pub const fn direction(self) -> TextSelectionDirection {
        self.direction
    }
}

#[cfg(test)]
mod tests {
    use super::{TextSelection, TextSelectionDirection};

    #[test]
    fn selection_ordering_is_validated_and_direction_is_preserved() {
        assert!(TextSelection::new(3, 2, TextSelectionDirection::None).is_err());
        let selection = TextSelection::new(1, 4, TextSelectionDirection::Backward)
            .expect("selection ordering is valid");
        assert_eq!(selection.start(), 1);
        assert_eq!(selection.end(), 4);
        assert_eq!(selection.direction(), TextSelectionDirection::Backward);
        assert_eq!(TextSelectionDirection::Other.as_str(), "other");
    }

    #[test]
    fn browser_direction_values_fail_closed_to_other() {
        assert_eq!(
            TextSelectionDirection::from_browser(Some("future")),
            TextSelectionDirection::Other
        );
        assert_eq!(
            TextSelectionDirection::from_browser(None),
            TextSelectionDirection::None
        );
    }
}
