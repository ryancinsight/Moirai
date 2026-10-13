//! A fresh exclusively owned local WebView2 user-data directory.

use std::{
    io,
    os::windows::{ffi::OsStrExt, fs::MetadataExt},
    path::{Component, Path, PathBuf, Prefix},
};

use windows::Win32::Globalization::{CSTR_EQUAL, CompareStringOrdinal};
use windows::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

/// Extended Windows path capacity includes the terminating UTF16 unit.
/// https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
pub(crate) const MAX_PROFILE_PATH_UNITS: usize = 32_767 - 1;

pub(super) fn validate(folder: &Path) -> io::Result<PathBuf> {
    let units = folder.as_os_str().encode_wide();
    if units.clone().count() > MAX_PROFILE_PATH_UNITS
        || char::decode_utf16(units).any(|character| character.is_err() || character == Ok('\0'))
    {
        return Err(invalid(
            "WebView2 profile path exceeds its Unicode native representation bound",
        ));
    }
    if !folder.is_absolute() {
        return Err(invalid(
            "WebView2 profile requires an absolute local directory",
        ));
    }
    let mut components = folder.components();
    let Some(Component::Prefix(prefix)) = components.next() else {
        return Err(invalid("WebView2 profile requires a local drive directory"));
    };
    let mut plain = match prefix.kind() {
        Prefix::Disk(letter) | Prefix::VerbatimDisk(letter) => {
            PathBuf::from(format!("{}:\\", char::from(letter)))
        }
        _ => return Err(invalid("WebView2 profile requires a local drive directory")),
    };
    check_directory(&plain)?;
    for component in components {
        match component {
            Component::RootDir => {}
            Component::Normal(part) if part != "." && part != ".." => {
                plain.push(part);
                check_directory(&plain)?;
            }
            _ => return Err(invalid("WebView2 profile path contains a relative segment")),
        }
    }
    if std::fs::read_dir(&plain)?.next().transpose()?.is_some() {
        return Err(invalid(
            "WebView2 profile directory must be empty before creation",
        ));
    }
    Ok(plain)
}

fn check_directory(path: &Path) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT.0 != 0 {
        return Err(invalid("WebView2 profile path contains a reparse point"));
    }
    if !metadata.is_dir() {
        return Err(invalid("WebView2 profile target is not a directory"));
    }
    Ok(())
}

/// Canonical directory spelling can preserve input case. Windows ordinal
/// component comparison prevents differing case from hiding an overlap.
pub(super) fn reject_overlap(profile: &Path, package: &Path) -> io::Result<()> {
    let profile = profile.canonicalize()?;
    let package = package.canonicalize()?;
    if contains(&profile, &package)? || contains(&package, &profile)? {
        return Err(invalid(
            "WebView2 profile and mapped package folders must not overlap",
        ));
    }
    Ok(())
}

pub(crate) fn verify_effective(expected: &Path, actual: &Path) -> io::Result<()> {
    if !actual.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WebView2 effective user-data directory is not absolute",
        ));
    }
    let expected = expected.canonicalize()?;
    let actual = actual.canonicalize()?;
    if !contains(&expected, &actual)? || !contains(&actual, &expected)? {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "WebView2 runtime overrides the exclusively owned user-data directory",
        ));
    }
    Ok(())
}

fn contains(path: &Path, prefix: &Path) -> io::Result<bool> {
    let mut path = path.components();
    for component in prefix.components() {
        let Some(actual) = path.next() else {
            return Ok(false);
        };
        let expected = component.as_os_str().encode_wide().collect::<Vec<_>>();
        let actual = actual.as_os_str().encode_wide().collect::<Vec<_>>();
        if i32::try_from(expected.len()).is_err() || i32::try_from(actual.len()).is_err() {
            return Err(invalid(
                "WebView2 directory component exceeds native comparison bound",
            ));
        }
        // SAFETY: both owned UTF16 slices and explicit lengths live through
        // this synchronous comparison. No pointers escape and all lengths fit
        // the native signed representation checked above.
        let result = unsafe { CompareStringOrdinal(&expected, &actual, true) };
        if result.0 == 0 {
            return Err(io::Error::other(format!(
                "WebView2 directory comparison: {}",
                windows::core::Error::from_thread()
            )));
        }
        if result != CSTR_EQUAL {
            return Ok(false);
        }
    }
    Ok(true)
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests;
