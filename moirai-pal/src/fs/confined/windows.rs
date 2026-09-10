//! Windows directory-handle anchored file opening via NtCreateFile.

use ::windows::Wdk::Foundation::OBJECT_ATTRIBUTES;
use ::windows::Wdk::Storage::FileSystem::{
    FILE_DIRECTORY_FILE, FILE_NON_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_REPARSE_POINT,
    FILE_SYNCHRONOUS_IO_NONALERT, NtCreateFile,
};
use ::windows::Win32::Foundation::{
    HANDLE, NTSTATUS, OBJ_CASE_INSENSITIVE, RtlNtStatusToDosError, STATUS_SUCCESS, UNICODE_STRING,
};
use ::windows::Win32::Storage::FileSystem::{
    FILE_ATTRIBUTE_NORMAL, FILE_GENERIC_READ, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
};
use ::windows::Win32::System::IO::IO_STATUS_BLOCK;
use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::mem::size_of;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::path::Path;

const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

pub(super) fn open(root: &Path, relative: &Path) -> io::Result<File> {
    let root = open_directory(root)?;
    let mut directory = root;
    let mut components = relative.components().peekable();
    loop {
        let Some(component) = components.next() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "relative path has no final component",
            ));
        };
        let is_directory = components.peek().is_some();
        let opened = open_relative(&directory, component.as_os_str(), is_directory)?;
        let attributes = opened.metadata()?.file_attributes();
        if attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "path component {} is a reparse point",
                    Path::new(component.as_os_str()).display()
                ),
            ));
        }
        if is_directory {
            if !opened.metadata()?.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!(
                        "path component {} is not a directory",
                        Path::new(component.as_os_str()).display()
                    ),
                ));
            }
            directory = opened;
        } else {
            if !opened.metadata()?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("path {} is not a regular file", relative.display()),
                ));
            }
            return Ok(opened);
        }
    }
}

fn open_directory(root: &Path) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS);
    let directory = options.open(root)?;
    if directory.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "root path is a reparse point",
        ));
    }
    if !directory.metadata()?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            "root path is not a directory",
        ));
    }
    Ok(directory)
}

fn open_relative(parent: &File, component: &OsStr, directory: bool) -> io::Result<File> {
    let mut name: Vec<u16> = component.encode_wide().collect();
    let byte_length = name
        .len()
        .checked_mul(size_of::<u16>())
        .and_then(|length| u16::try_from(length).ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Windows path component exceeds the NT object-name limit",
            )
        })?;
    let unicode_name = UNICODE_STRING {
        Length: byte_length,
        MaximumLength: byte_length,
        // SAFETY: the buffer is kept alive for the synchronous system call
        // and contains exactly the UTF-16 units described above.
        Buffer: ::windows::core::PWSTR(name.as_mut_ptr()),
    };
    let attributes = OBJECT_ATTRIBUTES {
        Length: u32::try_from(size_of::<OBJECT_ATTRIBUTES>()).expect("OBJECT_ATTRIBUTES fits u32"),
        RootDirectory: HANDLE(parent.as_raw_handle()),
        ObjectName: &unicode_name,
        Attributes: OBJ_CASE_INSENSITIVE,
        SecurityDescriptor: std::ptr::null(),
        SecurityQualityOfService: std::ptr::null(),
    };
    let mut status_block = IO_STATUS_BLOCK::default();
    let mut handle = HANDLE::default();
    let options = FILE_OPEN_REPARSE_POINT
        | FILE_SYNCHRONOUS_IO_NONALERT
        | if directory {
            FILE_DIRECTORY_FILE
        } else {
            FILE_NON_DIRECTORY_FILE
        };
    // SAFETY: all pointers refer to stack or owned buffers that remain
    // valid for this synchronous call; parent is an open directory
    // handle, the access/share flags are read-only, and a successful
    // result transfers one unique handle to File below.
    let status = unsafe {
        NtCreateFile(
            &mut handle,
            FILE_GENERIC_READ,
            &attributes,
            &mut status_block,
            None,
            FILE_ATTRIBUTE_NORMAL,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            FILE_OPEN,
            options,
            None,
            0,
        )
    };
    if status != STATUS_SUCCESS {
        return Err(ntstatus_error(status));
    }
    if handle.is_invalid() {
        return Err(io::Error::other("NtCreateFile returned an invalid handle"));
    }
    // SAFETY: NtCreateFile returned success and a unique owned handle;
    // File takes responsibility for closing it exactly once.
    Ok(unsafe { File::from_raw_handle(handle.0) })
}

fn ntstatus_error(status: NTSTATUS) -> io::Error {
    // SAFETY: RtlNtStatusToDosError is a pure conversion of the status
    // returned by NtCreateFile and does not dereference Rust pointers.
    let code = unsafe { RtlNtStatusToDosError(status) };
    io::Error::from_raw_os_error(i32::from_ne_bytes(code.to_ne_bytes()))
}
