//! Validated save-dialog request and its Windows file-type encoding.

use std::io;

use windows::{Win32::UI::Shell::Common::COMDLG_FILTERSPEC, core::PCWSTR};

use super::{MAX_DIALOG_FILTERS, MAX_DIALOG_PATH_UNITS};

/// One entry of a save dialog's file-type list.
///
/// The label is the text shown to the user, such as `PNG image`. The pattern
/// is a Windows file-type pattern: one or more wildcard names separated by
/// semicolons, such as `*.png` or `*.nii;*.nii.gz`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFilter {
    label: String,
    pattern: String,
}

impl FileFilter {
    /// Validates one file-type entry.
    ///
    /// # Examples
    /// ```
    /// use moirai_pal::windows::dialog::FileFilter;
    ///
    /// let filter = FileFilter::new("NIfTI volume", "*.nii;*.nii.gz")?;
    /// assert_eq!(filter.label(), "NIfTI volume");
    /// assert_eq!(filter.pattern(), "*.nii;*.nii.gz");
    /// assert_eq!(
    ///     FileFilter::new("PNG image", "dir\\*.png").unwrap_err().kind(),
    ///     std::io::ErrorKind::InvalidInput
    /// );
    /// # Ok::<(), std::io::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `InvalidInput` when the label or pattern is empty, contains NUL,
    /// exceeds [`MAX_DIALOG_PATH_UNITS`] UTF-16 units, or when the pattern holds
    /// an empty or blank segment, a slash, a backslash or a colon.
    pub fn new(label: impl Into<String>, pattern: impl Into<String>) -> io::Result<Self> {
        let label = label.into();
        let pattern = pattern.into();
        check_text(&label, "filter label")?;
        check_text(&pattern, "filter pattern")?;
        if label.is_empty() || pattern.is_empty() {
            return Err(invalid("filter label and pattern must not be empty"));
        }
        if pattern
            .split(';')
            .any(|segment| segment.trim().is_empty() || segment.contains(['\\', '/', ':']))
        {
            return Err(invalid(
                "filter pattern must be non-blank wildcard names separated by semicolons",
            ));
        }
        Ok(Self { label, pattern })
    }

    /// Text shown to the user for this file type.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Wildcard names that select files of this type.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

/// Parameters of a native Save As dialog.
///
/// A dialog built from this request offers the suggested file name, lists the
/// filters in order with the first selected, and asks the user to confirm
/// before an existing file is chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveRequest {
    file_name: String,
    filters: Vec<FileFilter>,
}

impl SaveRequest {
    /// Validates a save request.
    ///
    /// An empty `file_name` suggests nothing, and an empty filter list offers
    /// every file.
    ///
    /// # Examples
    /// ```
    /// use moirai_pal::windows::dialog::{FileFilter, SaveRequest};
    ///
    /// let png = FileFilter::new("PNG image", "*.png")?;
    /// let request = SaveRequest::new("scan.png", [png.clone()])?;
    /// assert_eq!(request.file_name(), "scan.png");
    /// assert_eq!(request.filters(), [png]);
    /// assert_eq!(
    ///     SaveRequest::new("dir\\scan.png", []).unwrap_err().kind(),
    ///     std::io::ErrorKind::InvalidInput
    /// );
    /// # Ok::<(), std::io::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `InvalidInput` when the name contains NUL, a slash, a backslash
    /// or a colon, exceeds [`MAX_DIALOG_PATH_UNITS`] UTF-16 units, or when more
    /// than [`MAX_DIALOG_FILTERS`] filters follow.
    pub fn new(
        file_name: impl Into<String>,
        filters: impl IntoIterator<Item = FileFilter>,
    ) -> io::Result<Self> {
        let file_name = file_name.into();
        check_text(&file_name, "suggested file name")?;
        if file_name.contains(['\\', '/', ':']) {
            return Err(invalid(
                "suggested file name must not contain a path separator",
            ));
        }
        let filters: Vec<FileFilter> = filters.into_iter().take(MAX_DIALOG_FILTERS + 1).collect();
        if filters.len() > MAX_DIALOG_FILTERS {
            return Err(invalid("save request exceeds the bounded filter count"));
        }
        Ok(Self { file_name, filters })
    }

    /// Name the dialog proposes; empty when it proposes none.
    #[must_use]
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// File types in display order; the first is selected initially.
    #[must_use]
    pub fn filters(&self) -> &[FileFilter] {
        &self.filters
    }
}

/// NUL-terminated UTF-16 label and pattern buffers of a filter list.
///
/// `COMDLG_FILTERSPEC` holds raw pointers into these buffers. The pointers from
/// [`EncodedFilters::specs`] are valid while `self` lives and are needed only
/// for the `SetFileTypes` call, which copies the strings.
pub(super) struct EncodedFilters {
    entries: Vec<[Vec<u16>; 2]>,
}

impl EncodedFilters {
    pub(super) fn new(filters: &[FileFilter]) -> Self {
        Self {
            entries: filters
                .iter()
                .map(|filter| [wide(filter.label()), wide(filter.pattern())])
                .collect(),
        }
    }

    /// Specifications whose pointers are valid while `self` lives.
    pub(super) fn specs(&self) -> Vec<COMDLG_FILTERSPEC> {
        self.entries
            .iter()
            .map(|[label, pattern]| COMDLG_FILTERSPEC {
                pszName: PCWSTR(label.as_ptr()),
                pszSpec: PCWSTR(pattern.as_ptr()),
            })
            .collect()
    }
}

/// UTF-16 encoding of `text` followed by the terminating NUL.
pub(super) fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

fn check_text(text: &str, what: &str) -> io::Result<()> {
    if text.contains('\0') {
        return Err(invalid_text(what, "contains NUL"));
    }
    if text.encode_utf16().count() > MAX_DIALOG_PATH_UNITS {
        return Err(invalid_text(what, "exceeds the bounded UTF-16 limit"));
    }
    Ok(())
}

fn invalid_text(what: &str, reason: &str) -> io::Error {
    invalid(&format!("{what} {reason}"))
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: PCWSTR) -> Vec<u16> {
        // SAFETY: every pointer under test comes from a NUL-terminated buffer
        // owned by the `EncodedFilters` that outlives this read.
        unsafe { text.as_wide() }.to_vec()
    }

    fn units(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    #[test]
    fn filters_encode_to_nul_terminated_utf16_in_order() {
        let filters = [
            FileFilter::new("PNG image", "*.png").expect("valid filter"),
            FileFilter::new("NIfTI volume \u{e9}", "*.nii;*.nii.gz").expect("valid filter"),
        ];
        let encoded = EncodedFilters::new(&filters);
        let specs = encoded.specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(read(specs[0].pszName), units("PNG image"));
        assert_eq!(read(specs[0].pszSpec), units("*.png"));
        assert_eq!(read(specs[1].pszName), units("NIfTI volume \u{e9}"));
        assert_eq!(read(specs[1].pszSpec), units("*.nii;*.nii.gz"));
        assert_eq!(wide("A\u{1f600}"), vec![0x41, 0xd83d, 0xde00, 0]);
    }

    #[test]
    fn an_empty_filter_list_encodes_to_no_specs() {
        assert!(EncodedFilters::new(&[]).specs().is_empty());
    }

    #[test]
    fn filter_validation_rejects_each_malformed_part() {
        let long = "a".repeat(MAX_DIALOG_PATH_UNITS + 1);
        for (label, pattern) in [
            ("", "*.png"),
            ("PNG", ""),
            ("P\0NG", "*.png"),
            ("PNG", "*.p\0ng"),
            ("PNG", "*.png;"),
            ("PNG", ";*.png"),
            ("PNG", "dir\\*.png"),
            ("PNG", "dir/*.png"),
            ("PNG", "*.a;;*.b"),
            ("PNG", " "),
            ("PNG", "*.png; "),
            ("PNG", "*.png;dir\\*.png"),
            ("PNG", "C:*.png"),
            (long.as_str(), "*.png"),
            ("PNG", long.as_str()),
        ] {
            let error = FileFilter::new(label, pattern).expect_err("malformed filter");
            assert_eq!(
                error.kind(),
                io::ErrorKind::InvalidInput,
                "{label:?} {pattern:?}"
            );
        }
        let filter = FileFilter::new("Any", "*.*").expect("wildcard pattern");
        assert_eq!((filter.label(), filter.pattern()), ("Any", "*.*"));
    }

    #[test]
    fn the_text_bound_counts_utf16_units_not_bytes_or_scalars() {
        let at_bound = [
            "a".repeat(MAX_DIALOG_PATH_UNITS),
            // 20,000 units occupy 40,000 UTF-8 bytes, above the bound in bytes.
            "\u{e9}".repeat(20_000),
        ];
        // 16,384 scalars are 32,768 units (65,536 UTF-8 bytes): above the
        // bound in units, though their scalar count is far below it.
        let above_bound = "\u{1f600}".repeat(16_384);
        assert_eq!(
            above_bound.encode_utf16().count(),
            MAX_DIALOG_PATH_UNITS + 1
        );
        for text in &at_bound {
            let request = SaveRequest::new(text.as_str(), []).expect("name within the bound");
            assert_eq!(request.file_name(), text);
            let labelled = FileFilter::new(text.as_str(), "*.png").expect("label within the bound");
            assert_eq!(labelled.label(), text);
            let patterned =
                FileFilter::new("PNG", text.as_str()).expect("pattern within the bound");
            assert_eq!(patterned.pattern(), text);
        }
        for result in [
            SaveRequest::new(above_bound.as_str(), []).map(drop),
            FileFilter::new(above_bound.as_str(), "*.png").map(drop),
            FileFilter::new("PNG", above_bound.as_str()).map(drop),
        ] {
            assert_eq!(
                result.expect_err("text above the bound").kind(),
                io::ErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn request_validation_rejects_malformed_suggested_names() {
        for name in ["a\0b", "dir\\scan.png", "dir/scan.png", "C:scan.png"] {
            let error = SaveRequest::new(name, []).expect_err("malformed name");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{name:?}");
        }
        let unnamed = SaveRequest::new("", []).expect("an empty name suggests nothing");
        assert_eq!(unnamed.file_name(), "");
        assert!(unnamed.filters().is_empty());
    }

    #[test]
    fn a_request_keeps_its_filters_in_the_order_given() {
        let png = FileFilter::new("PNG image", "*.png").expect("valid filter");
        let nifti = FileFilter::new("NIfTI volume", "*.nii;*.nii.gz").expect("valid filter");
        let request =
            SaveRequest::new("scan", [png.clone(), nifti.clone()]).expect("valid request");
        assert_eq!(request.file_name(), "scan");
        assert_eq!(request.filters(), [png.clone(), nifti.clone()]);
        let reversed = SaveRequest::new("scan", [nifti.clone(), png.clone()]).expect("valid");
        assert_eq!(reversed.filters(), [nifti, png]);
    }

    #[test]
    fn the_filter_count_is_bounded_before_the_native_call() {
        let filter = FileFilter::new("Any", "*.*").expect("valid filter");
        let at_bound = SaveRequest::new("", vec![filter.clone(); MAX_DIALOG_FILTERS])
            .expect("count at the bound");
        assert_eq!(at_bound.filters().len(), MAX_DIALOG_FILTERS);
        for filters in [
            vec![filter.clone(); MAX_DIALOG_FILTERS + 1],
            vec![filter.clone(); 2 * MAX_DIALOG_FILTERS],
        ] {
            let error = SaveRequest::new("", filters).expect_err("count above the bound");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
        // An unbounded source is cut off at the bound instead of collected.
        let error = SaveRequest::new("", std::iter::repeat(filter)).expect_err("endless filters");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(u32::try_from(MAX_DIALOG_FILTERS), Ok(1_024));
    }
}
