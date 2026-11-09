//! `Show` is modal and cannot run headless, so these tests cover everything
//! around it: option flags as pure values, and a real dialog object that is
//! configured and read back but never shown. The user's interaction (picking,
//! confirming an overwrite, cancelling) needs a manual run.

use super::*;
use windows::Win32::Foundation::{E_FAIL, E_NOINTERFACE};
use windows::Win32::System::Com::CoTaskMemAlloc;
use windows::Win32::UI::Shell::{FOS_NOCHANGEDIR, IFileOpenDialog, IFileSaveDialog};
use windows::core::Interface;

fn save_request() -> SaveRequest {
    SaveRequest::new(
        "scan.png",
        [
            FileFilter::new("PNG image", "*.png").expect("valid filter"),
            FileFilter::new("NIfTI volume", "*.nii;*.nii.gz").expect("valid filter"),
        ],
    )
    .expect("valid request")
}

#[test]
fn selection_kinds_are_explicit() {
    assert_ne!(DialogSelection::File, DialogSelection::Folder);
    assert_ne!(DialogSelection::File, DialogSelection::Save(save_request()));
    assert_ne!(
        DialogSelection::Save(save_request()),
        DialogSelection::Save(SaveRequest::new("other.png", []).expect("valid request"))
    );
}

#[test]
fn option_flags_follow_the_selection_and_keep_the_starting_set() {
    let none = FILEOPENDIALOGOPTIONS(0);
    assert_eq!(
        dialog_options(none, &DialogSelection::File),
        FOS_FORCEFILESYSTEM
    );
    assert_eq!(
        dialog_options(none, &DialogSelection::Folder),
        FOS_FORCEFILESYSTEM | FOS_PICKFOLDERS
    );
    assert_eq!(
        dialog_options(none, &DialogSelection::Save(save_request())),
        FOS_FORCEFILESYSTEM | FOS_OVERWRITEPROMPT | FOS_PATHMUSTEXIST
    );
    for selection in [
        DialogSelection::File,
        DialogSelection::Folder,
        DialogSelection::Save(save_request()),
    ] {
        let starting = dialog_options(none, &selection);
        assert_eq!(
            dialog_options(FOS_NOCHANGEDIR, &selection),
            FOS_NOCHANGEDIR | starting,
            "{selection:?} keeps the starting set"
        );
    }
}

#[test]
fn a_configured_save_dialog_reports_the_request_it_was_built_from() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let dialog = configure(&DialogSelection::Save(save_request())).expect("save dialog");
    // SAFETY: the dialog is a live COM interface owned by this test.
    let options = unsafe { dialog.GetOptions() }.expect("dialog options");
    for flag in [FOS_FORCEFILESYSTEM, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST] {
        assert_eq!(options.0 & flag.0, flag.0, "{flag:?} is set");
    }
    assert_eq!(options.0 & FOS_PICKFOLDERS.0, 0);
    // SAFETY: as above; the returned buffer is freed by `TaskMemoryPath`.
    let name = unsafe { dialog.GetFileName() }.expect("suggested name");
    assert_eq!(
        TaskMemoryPath::new(name).to_path().expect("valid name"),
        PathBuf::from("scan.png")
    );
    // SAFETY: as above.
    let index = unsafe { dialog.GetFileTypeIndex() }.expect("file type index");
    assert_eq!(index, 1, "the first filter is selected");
}

#[test]
fn save_selections_build_a_save_dialog_and_the_others_an_open_dialog() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let save = configure(&DialogSelection::Save(save_request())).expect("save dialog");
    let save_interface = save
        .cast::<IFileSaveDialog>()
        .expect("invariant: a dialog from the FileSaveDialog class implements IFileSaveDialog");
    // SAFETY: the interface is live and owned by this test.
    let options = unsafe { save_interface.GetOptions() }.expect("save options");
    assert_eq!(
        options.0 & FOS_FORCEFILESYSTEM.0,
        FOS_FORCEFILESYSTEM.0,
        "the save interface reports the options the selection configured"
    );
    assert_eq!(
        save.cast::<IFileOpenDialog>()
            .expect_err("a save dialog is not an open dialog")
            .code(),
        E_NOINTERFACE
    );
    for selection in [DialogSelection::File, DialogSelection::Folder] {
        let open = configure(&selection).expect("open dialog");
        let open_interface = open
            .cast::<IFileOpenDialog>()
            .expect("invariant: a dialog from the FileOpenDialog class implements IFileOpenDialog");
        // SAFETY: the interface is live and owned by this test.
        let options = unsafe { open_interface.GetOptions() }.expect("open options");
        assert_eq!(
            options.0 & FOS_PICKFOLDERS.0 != 0,
            selection == DialogSelection::Folder,
            "{selection:?}: the open interface reports the options of its selection"
        );
        assert_eq!(
            open.cast::<IFileSaveDialog>()
                .expect_err("an open dialog is not a save dialog")
                .code(),
            E_NOINTERFACE,
            "{selection:?}"
        );
    }
}

#[test]
fn a_save_dialog_without_filters_configures_and_keeps_its_name() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let request = SaveRequest::new("scan.png", []).expect("valid request");
    let dialog = configure(&DialogSelection::Save(request))
        .expect("an empty filter list must not reach SetFileTypeIndex");
    // SAFETY: the dialog is a live COM interface owned by this test.
    let name = unsafe { dialog.GetFileName() }.expect("suggested name");
    assert_eq!(
        TaskMemoryPath::new(name).to_path().expect("valid name"),
        PathBuf::from("scan.png")
    );
}

/// `SetFileTypeIndex` returns S_OK for any index and `GetFileTypeIndex`
/// clamps the stored index to the length of the file-type list, so the
/// read-back after an index beyond the list is that list's length.
fn index_read_back_after_setting(dialog: &IFileDialog, beyond: u32) -> u32 {
    // SAFETY: `dialog` is a live COM interface borrowed for this call.
    unsafe { dialog.SetFileTypeIndex(beyond) }.expect("set file type index");
    // SAFETY: as above.
    unsafe { dialog.GetFileTypeIndex() }.expect("file type index")
}

#[test]
fn the_native_list_holds_exactly_the_filters_of_the_request() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let dialog = configure(&DialogSelection::Save(save_request())).expect("save dialog");
    assert_eq!(
        index_read_back_after_setting(&dialog, 2),
        2,
        "both filters are present"
    );
    assert_eq!(
        index_read_back_after_setting(&dialog, 3),
        2,
        "no filter beyond the request is present"
    );
}

#[test]
fn a_request_at_every_bound_configures_and_reads_back() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let name = "a".repeat(MAX_DIALOG_PATH_UNITS);
    let filter = FileFilter::new("Any", "*.*").expect("valid filter");
    let request = SaveRequest::new(name.as_str(), vec![filter; MAX_DIALOG_FILTERS])
        .expect("request at the bounds");
    let dialog = configure(&DialogSelection::Save(request)).expect("save dialog");
    // SAFETY: the dialog is a live COM interface owned by this test.
    let stored = unsafe { dialog.GetFileName() }.expect("suggested name");
    assert_eq!(
        TaskMemoryPath::new(stored).to_path().expect("valid name"),
        PathBuf::from(&name)
    );
    let count = u32::try_from(MAX_DIALOG_FILTERS).expect("the bound fits u32");
    assert_eq!(index_read_back_after_setting(&dialog, count), count);
    assert_eq!(index_read_back_after_setting(&dialog, count + 1), count);
}

#[test]
fn a_dialog_without_a_suggested_name_configures() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let png = FileFilter::new("PNG image", "*.png").expect("valid filter");
    let request = SaveRequest::new("", [png]).expect("valid request");
    let dialog = configure(&DialogSelection::Save(request)).expect("save dialog");
    // SAFETY: the dialog is a live COM interface owned by this test.
    let index = unsafe { dialog.GetFileTypeIndex() }.expect("file type index");
    assert_eq!(index, 1);
    // SAFETY: as above. A dialog that never received a name reports
    // E_FAIL with a null buffer, so a name set to "" would not.
    let name = unsafe { dialog.GetFileName() };
    assert_eq!(name.expect_err("no suggested name was set").code(), E_FAIL);
}

#[test]
fn a_dialog_copies_the_suggested_name_it_is_given() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let dialog = configure(&DialogSelection::File).expect("open dialog");
    let mut name = wide("kept.png");
    // SAFETY: `name` is NUL-terminated and valid for this call.
    unsafe { dialog.SetFileName(PCWSTR(name.as_ptr())) }.expect("set name");
    name.fill(u16::from(b'X'));
    // SAFETY: the dialog is a live COM interface owned by this test.
    let stored = unsafe { dialog.GetFileName() }.expect("suggested name");
    assert_eq!(
        TaskMemoryPath::new(stored).to_path().expect("valid name"),
        PathBuf::from("kept.png"),
        "the dialog does not read the caller's buffer after the call"
    );
}

#[test]
fn a_configured_folder_dialog_does_not_confirm_overwrites() {
    let _apartment = ComApartment::initialize().expect("COM apartment");
    let dialog = configure(&DialogSelection::Folder).expect("folder dialog");
    // SAFETY: the dialog is a live COM interface owned by this test.
    let options = unsafe { dialog.GetOptions() }.expect("dialog options");
    assert_eq!(options.0 & FOS_PICKFOLDERS.0, FOS_PICKFOLDERS.0);
    assert_eq!(options.0 & FOS_OVERWRITEPROMPT.0, 0);
}

#[test]
fn a_task_memory_path_decodes_to_the_buffer_it_owns() {
    let expected = PathBuf::from("C:\\Moirai\\\u{e9}t\u{e9}.txt");
    let units: Vec<u16> = expected
        .to_str()
        .expect("the fixture is UTF-8")
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: the allocation holds `units.len()` code units, is written in
    // full before use, and is handed to `TaskMemoryPath`, which frees it
    // with `CoTaskMemFree`.
    let buffer = unsafe { CoTaskMemAlloc(units.len() * size_of::<u16>()) }.cast::<u16>();
    assert!(!buffer.is_null());
    // SAFETY: `buffer` has room for `units.len()` code units.
    unsafe { std::ptr::copy_nonoverlapping(units.as_ptr(), buffer, units.len()) };
    let owner = TaskMemoryPath::new(PWSTR(buffer));
    assert_eq!(owner.to_path().expect("valid path"), expected);
    assert_eq!(
        owner
            .to_path()
            .expect("decoding leaves the buffer readable"),
        expected
    );
}

#[test]
fn cancellation_uses_the_windows_hresult() {
    assert_eq!(
        HRESULT::from_win32(ERROR_CANCELLED.0).0.cast_unsigned(),
        0x8007_04c7
    );
}
