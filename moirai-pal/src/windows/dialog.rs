//! Bounded Windows common file, folder and save-destination selection.

use std::{io, path::PathBuf, slice};

mod save;

use save::{EncodedFilters, wide};
pub use save::{FileFilter, SaveRequest};

use windows::{
    Win32::{
        Foundation::ERROR_CANCELLED,
        Globalization::lstrlenW,
        System::Com::{
            CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx,
            CoTaskMemFree, CoUninitialize,
        },
        UI::Shell::{
            FILEOPENDIALOGOPTIONS, FOS_FORCEFILESYSTEM, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST,
            FOS_PICKFOLDERS, FileOpenDialog, FileSaveDialog, IFileDialog, SIGDN_FILESYSPATH,
        },
    },
    core::{HRESULT, PCWSTR, PWSTR},
};

/// Maximum UTF-16 code units accepted in one dialog text.
///
/// Bounds a path returned by the dialog, a suggested file name, and a filter
/// label or pattern. It is the Windows extended path limit.
pub const MAX_DIALOG_PATH_UNITS: usize = 32_767;

/// Maximum file-type entries in one save dialog.
///
/// The native list is a combo box the user reads; no documented limit exists,
/// so the bound is a resource limit far above any list of export formats. It
/// also keeps the count inside the `u32` that `SetFileTypes` takes.
pub const MAX_DIALOG_FILTERS: usize = 1_024;

/// Kind of native filesystem object chosen through [`pick`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DialogSelection {
    /// Select one existing regular file.
    File,
    /// Select one directory containing application input files.
    Folder,
    /// Choose the destination path of a file to write, which need not exist.
    Save(SaveRequest),
}

/// Opens the Windows common dialog and returns the user's filesystem selection.
///
/// Cancellation is an expected result and returns `Ok(None)`. The returned path
/// is only a user selection; callers still own validation, authorization and
/// bounded reading or writing. [`DialogSelection::Save`] returns the chosen
/// destination after the dialog has asked the user to confirm replacing an
/// existing file; the caller creates or writes the file.
/// WebAssembly and non-Windows targets do not expose this module, so browser
/// consumers use the DOM file provider instead.
///
/// # Errors
/// Returns an operating-system or COM error when the dialog cannot be created
/// or configured (options, suggested name, file types), shown, or when its
/// result cannot be converted to a bounded UTF-8 path.
pub fn pick(selection: DialogSelection) -> io::Result<Option<PathBuf>> {
    let _apartment = ComApartment::initialize()?;
    let dialog = configure(&selection)?;

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
    TaskMemoryPath::new(raw_path).to_path().map(Some)
}

/// Creates the dialog for `selection` and applies everything but `Show`.
///
/// The calling thread must hold an initialized COM apartment.
fn configure(selection: &DialogSelection) -> io::Result<IFileDialog> {
    let class = match selection {
        DialogSelection::Save(_) => &FileSaveDialog,
        DialogSelection::File | DialogSelection::Folder => &FileOpenDialog,
    };
    // SAFETY: COM is initialized on this thread. Releasing the returned
    // interface before the apartment guard drops is the caller's obligation:
    // `pick` holds the guard and the interface in one scope, and the tests
    // declare the guard first.
    let dialog: IFileDialog =
        unsafe { CoCreateInstance(class, None, CLSCTX_INPROC_SERVER) }.map_err(windows_error)?;

    // SAFETY: `dialog` is a live COM interface owned by this scope.
    let base = unsafe { dialog.GetOptions() }.map_err(windows_error)?;
    // SAFETY: the option value contains only documented common-dialog flags.
    unsafe { dialog.SetOptions(dialog_options(base, selection)) }.map_err(windows_error)?;

    if let DialogSelection::Save(request) = selection {
        apply_save_request(&dialog, request)?;
    }
    Ok(dialog)
}

/// Options for `selection` on top of those the dialog starts with.
///
/// Save adds the overwrite confirmation and requires the parent directory of
/// the chosen path to exist.
fn dialog_options(
    base: FILEOPENDIALOGOPTIONS,
    selection: &DialogSelection,
) -> FILEOPENDIALOGOPTIONS {
    let options = base | FOS_FORCEFILESYSTEM;
    match selection {
        DialogSelection::File => options,
        DialogSelection::Folder => options | FOS_PICKFOLDERS,
        DialogSelection::Save(_) => options | FOS_OVERWRITEPROMPT | FOS_PATHMUSTEXIST,
    }
}

fn apply_save_request(dialog: &IFileDialog, request: &SaveRequest) -> io::Result<()> {
    if !request.file_name().is_empty() {
        let name = wide(request.file_name());
        // SAFETY: `name` is a NUL-terminated UTF-16 buffer that stays valid
        // for the duration of this call.
        unsafe { dialog.SetFileName(PCWSTR(name.as_ptr())) }.map_err(windows_error)?;
    }
    // `SetFileTypeIndex` fails with E_FAIL while no file types are set, so an
    // empty list leaves both calls out.
    if !request.filters().is_empty() {
        let encoded = EncodedFilters::new(request.filters());
        let specs = encoded.specs();
        // SAFETY: every specification points into `encoded`, which stays valid
        // for the duration of this call.
        unsafe { dialog.SetFileTypes(&specs) }.map_err(windows_error)?;
        // SAFETY: `dialog` is a live COM interface borrowed for this call.
        // File-type indices are one-based and the list is non-empty.
        unsafe { dialog.SetFileTypeIndex(1) }.map_err(windows_error)?;
    }
    Ok(())
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

    /// Decodes the buffer. The owner keeps the pointer, so `Drop` releases the
    /// task memory on the success path as on every error path.
    fn to_path(&self) -> io::Result<PathBuf> {
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
mod tests;
