//! Bounded Windows common file and folder selection.

use std::{io, path::PathBuf, slice};

use windows::{
    Win32::{
        Foundation::ERROR_CANCELLED,
        Globalization::lstrlenW,
        System::Com::{
            CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx,
            CoTaskMemFree, CoUninitialize,
        },
        UI::Shell::{
            FOS_FORCEFILESYSTEM, FOS_PICKFOLDERS, FileOpenDialog, IFileOpenDialog,
            SIGDN_FILESYSPATH,
        },
    },
    core::{HRESULT, PCWSTR, PWSTR},
};

/// Maximum UTF-16 code units accepted from one common-dialog path.
pub const MAX_DIALOG_PATH_UNITS: usize = 32_767;

/// Kind of native filesystem object returned by [`pick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogSelection {
    /// Select one regular file.
    File,
    /// Select one directory containing application input files.
    Folder,
}

/// Opens the Windows common dialog and returns the user's filesystem selection.
///
/// Cancellation is an expected result and returns `Ok(None)`. The returned path
/// is only a user selection; callers still own validation, authorization and
/// bounded reading. WebAssembly and non-Windows targets do not expose this
/// module, so browser consumers use the DOM file provider instead.
///
/// # Errors
/// Returns an operating-system or COM error when the dialog cannot be created,
/// shown or converted to a bounded UTF-8 path.
pub fn pick(selection: DialogSelection) -> io::Result<Option<PathBuf>> {
    let _apartment = ComApartment::initialize()?;
    // SAFETY: COM is initialized on this thread and the dialog interface is
    // released before the apartment guard leaves scope.
    let dialog: IFileOpenDialog =
        unsafe { CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER) }
            .map_err(windows_error)?;

    let mut options = unsafe { dialog.GetOptions() }.map_err(windows_error)?;
    options |= FOS_FORCEFILESYSTEM;
    if selection == DialogSelection::Folder {
        options |= FOS_PICKFOLDERS;
    }
    // SAFETY: `dialog` is a live COM interface owned by this scope and the
    // option value contains only documented common-dialog flags.
    unsafe { dialog.SetOptions(options) }.map_err(windows_error)?;

    // SAFETY: the modal dialog retains no owner handle or Rust pointer after
    // this synchronous call returns.
    let shown = unsafe { dialog.Show(None) };
    if let Err(error) = shown {
        if error.code() == HRESULT::from_win32(ERROR_CANCELLED.0) {
            return Ok(None);
        }
        return Err(windows_error(error));
    }

    // SAFETY: `dialog` remains live after a successful modal result and the
    // returned shell item owns its own COM reference.
    let item = unsafe { dialog.GetResult() }.map_err(windows_error)?;
    // SAFETY: the shell item returns a task-memory UTF-16 buffer which is
    // immediately wrapped by `TaskMemoryPath` and freed on every exit path.
    let raw_path = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH) }.map_err(windows_error)?;
    TaskMemoryPath::new(raw_path).into_path().map(Some)
}

struct ComApartment {
    initialized: bool,
}

impl ComApartment {
    fn initialize() -> io::Result<Self> {
        // SAFETY: this call changes only the COM apartment of the current
        // thread; the matching uninitialization is guarded by `Drop`.
        let result = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        if result.is_ok() {
            Ok(Self { initialized: true })
        } else {
            Err(windows_error(result.into()))
        }
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.initialized {
            // SAFETY: the guard is dropped on the same thread after every COM
            // interface created by `pick` has been released.
            unsafe { CoUninitialize() };
        }
    }
}

struct TaskMemoryPath(PWSTR);

impl TaskMemoryPath {
    fn new(value: PWSTR) -> Self {
        Self(value)
    }

    fn into_path(mut self) -> io::Result<PathBuf> {
        if self.0.0.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Windows common dialog returned a null path",
            ));
        }
        // SAFETY: the dialog contract returns a NUL-terminated task-memory
        // buffer; `lstrlenW` reads only that buffer and retains no pointer.
        let length = unsafe { lstrlenW(PCWSTR(self.0.0)) };
        if length < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Windows common dialog returned a negative path length",
            ));
        }
        let length = usize::try_from(length).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Windows common dialog path length overflowed usize",
            )
        })?;
        if length > MAX_DIALOG_PATH_UNITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Windows common dialog path exceeds the bounded UTF-16 limit",
            ));
        }
        // SAFETY: `length` comes from the NUL-terminated buffer above and the
        // task-memory owner remains alive until this method returns.
        let units = unsafe { slice::from_raw_parts(self.0.0, length) };
        let mut value = String::new();
        value
            .try_reserve_exact(length)
            .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "path allocation failed"))?;
        for character in char::decode_utf16(units.iter().copied()) {
            value.push(character.map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Windows common dialog path contains invalid UTF-16",
                )
            })?);
        }
        self.0 = PWSTR::null();
        Ok(PathBuf::from(value))
    }
}

impl Drop for TaskMemoryPath {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            // SAFETY: the pointer came from `IShellItem::GetDisplayName`, which
            // specifies task-memory ownership for `CoTaskMemFree`.
            unsafe { CoTaskMemFree(Some(self.0.0.cast())) };
        }
    }
}

fn windows_error(error: windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(error.code().0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_kinds_are_explicit() {
        assert_ne!(DialogSelection::File, DialogSelection::Folder);
    }

    #[test]
    fn cancellation_uses_the_windows_hresult() {
        assert_eq!(HRESULT::from_win32(ERROR_CANCELLED.0).0 as u32, 0x8007_04c7);
    }
}
