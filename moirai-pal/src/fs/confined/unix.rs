//! Unix directory-handle anchored file opening via openat.

use super::*;
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;

pub(super) fn open(root: &Path, relative: &Path) -> io::Result<File> {
    let directory = open_directory(root)?;
    let mut directory = directory;
    let mut components = relative.components().peekable();
    let file = loop {
        let Some(component) = components.next() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "relative path has no final component",
            ));
        };
        let component_name = component.as_os_str();
        let name = CString::new(component_name.as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "path component {} contains NUL",
                    Path::new(component_name).display()
                ),
            )
        })?;
        let flags = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW;
        // SAFETY: directory is an open directory descriptor owned by
        // this function, name is NUL-terminated, and the flags request
        // neither creation nor mutation. A successful descriptor is
        // transferred exactly once into File below.
        let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            let error = io::Error::last_os_error().with_context(|| {
                format!(
                    "failed to open path component {} below {}",
                    Path::new(component_name).display(),
                    root.display()
                )
            });
            return Err(error);
        }
        // SAFETY: fd is the unique descriptor returned by openat; the
        // resulting File becomes its sole owner and closes it on every
        // return path.
        let opened = unsafe { File::from_raw_fd(fd) };
        if components.peek().is_some() {
            if !opened.metadata()?.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!(
                        "path component {} is not a directory",
                        Path::new(component_name).display()
                    ),
                ));
            }
            directory = opened;
        } else {
            break opened;
        }
    };

    if !file.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path {} is not a regular file", relative.display()),
        ));
    }
    Ok(file)
}

fn open_directory(root: &Path) -> io::Result<File> {
    let root = CString::new(root.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("root path {} contains NUL", root.display()),
        )
    })?;
    let flags = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW;
    // SAFETY: root is a NUL-terminated path and the flags request a
    // read-only handle without following the final link. The descriptor
    // is transferred to File immediately.
    let fd = unsafe { libc::open(root.as_ptr(), flags) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd is newly returned by open and is transferred to the
    // returned File, which closes it exactly once.
    let directory = unsafe { File::from_raw_fd(fd) };
    if !directory.metadata()?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            "root path is not a directory",
        ));
    }
    Ok(directory)
}

trait IoContext {
    fn with_context<F>(self, context: F) -> Self
    where
        F: FnOnce() -> String;
}

impl IoContext for io::Error {
    fn with_context<F>(self, context: F) -> Self
    where
        F: FnOnce() -> String,
    {
        let message = format!("{}: {}", context(), self);
        if self.raw_os_error().is_some() {
            self
        } else {
            io::Error::new(self.kind(), message)
        }
    }
}
