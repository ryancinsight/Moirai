//! Directory-handle anchored native file opening.
//!
//! Path canonicalization followed by a second path open leaves a
//! time-of-check/time-of-use race. This module validates the lexical path and
//! resolves every component from an owned directory handle instead.

use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Component, PathBuf};

/// Open a regular file below root without following links in its path.
///
/// Path may be absolute below root or relative to root. Absolute and parent
/// components are rejected, as are the root itself and non-normal
/// components. The returned handle is the object opened during the confined
/// walk; callers must read that handle rather than reopening path.
///
/// Native providers use directory-relative operating-system calls. Browser
/// targets do not have native path authority and return
/// io::ErrorKind::Unsupported; use the DOM file provider for browser file
/// entries.
///
/// # Errors
///
/// Returns an error when the root is not a directory, the candidate is
/// outside the root, a path component is a link or has the wrong kind, the
/// target is not a regular file, or the target platform has no native
/// filesystem provider.
pub fn open_file_within_root<P, R>(path: P, root: R) -> io::Result<File>
where
    P: AsRef<Path>,
    R: AsRef<Path>,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (path, root);
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "root-confined native file opening is unsupported on WebAssembly",
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let root = root.as_ref();
        let relative = relative_path(path.as_ref(), root)?;

        #[cfg(unix)]
        {
            unix::open(root, &relative)
        }

        #[cfg(windows)]
        {
            windows::open(root, &relative)
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = (root, relative);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "root-confined native file opening is unsupported on this target",
            ))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn relative_path(path: &Path, root: &Path) -> io::Result<PathBuf> {
    let relative = if path.is_absolute() {
        path.strip_prefix(root).map_err(|_| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("path {} is outside root {}", path.display(), root.display()),
            )
        })?
    } else {
        path
    };

    if relative.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path resolves to the root directory",
        ));
    }
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "path {} contains a non-normal component",
                relative.display()
            ),
        ));
    }
    Ok(relative.to_path_buf())
}

#[cfg(unix)]
mod unix {
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
            io::Error::new(self.kind(), format!("{}: {}", context(), self))
        }
    }
}

#[cfg(windows)]
mod windows {
    use super::*;
    use ::windows::Wdk::Foundation::OBJECT_ATTRIBUTES;
    use ::windows::Wdk::Storage::FileSystem::{
        FILE_DIRECTORY_FILE, FILE_NON_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_REPARSE_POINT,
        FILE_SYNCHRONOUS_IO_NONALERT, NtCreateFile,
    };
    use ::windows::Win32::Foundation::{
        HANDLE, NTSTATUS, OBJ_CASE_INSENSITIVE, RtlNtStatusToDosError, STATUS_SUCCESS,
        UNICODE_STRING,
    };
    use ::windows::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_NORMAL, FILE_GENERIC_READ, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE,
    };
    use ::windows::Win32::System::IO::IO_STATUS_BLOCK;
    use std::ffi::OsStr;
    use std::mem::size_of;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use std::os::windows::io::{AsRawHandle, FromRawHandle};

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
            Length: u32::try_from(size_of::<OBJECT_ATTRIBUTES>())
                .expect("OBJECT_ATTRIBUTES fits u32"),
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
}

#[cfg(test)]
mod tests {
    use super::open_file_within_root;
    use std::fs;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "moirai_pal_confined_{name}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    fn remove_tree(path: &PathBuf) {
        if path.exists() {
            fs::remove_dir_all(path).expect("test tree cleanup must succeed");
        }
    }

    #[test]
    fn opens_nested_regular_file_and_returns_source_handle() {
        let root = test_root("nested");
        let directory = root.join("series");
        let path = directory.join("slice.dcm");
        fs::create_dir_all(&directory).expect("test directory creation must succeed");
        fs::write(&path, b"dicom-bytes").expect("test file creation must succeed");

        let mut file = open_file_within_root(&path, &root).expect("confined open must succeed");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .expect("confined handle read must succeed");
        assert_eq!(bytes, b"dicom-bytes");

        remove_tree(&root);
    }

    #[test]
    fn rejects_parent_and_absolute_escape_paths() {
        let root = test_root("escape");
        fs::create_dir_all(&root).expect("test root creation must succeed");
        fs::write(root.join("slice.dcm"), b"safe").expect("test file creation must succeed");

        let parent = open_file_within_root(root.join("..").join("outside.dcm"), &root)
            .expect_err("parent traversal must be rejected");
        assert_eq!(parent.kind(), std::io::ErrorKind::InvalidInput);

        let outside = test_root("outside").join("slice.dcm");
        let absolute = open_file_within_root(&outside, &root)
            .expect_err("absolute path outside root must be rejected");
        assert_eq!(absolute.kind(), std::io::ErrorKind::PermissionDenied);

        remove_tree(&root);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_intermediate_and_final_symlinks() {
        use std::os::unix::fs::symlink;

        let root = test_root("links");
        let outside = test_root("link-target");
        fs::create_dir_all(&root).expect("test root creation must succeed");
        fs::create_dir_all(&outside).expect("test target creation must succeed");
        fs::write(outside.join("slice.dcm"), b"outside")
            .expect("target file creation must succeed");
        fs::write(root.join("real.dcm"), b"inside").expect("root file creation must succeed");
        symlink(&outside, root.join("nested")).expect("intermediate link creation must succeed");
        symlink(outside.join("slice.dcm"), root.join("final.dcm"))
            .expect("final link creation must succeed");

        let intermediate = open_file_within_root(root.join("nested/slice.dcm"), &root)
            .expect_err("intermediate symlink must be rejected");
        assert_eq!(intermediate.kind(), std::io::ErrorKind::Other);
        let final_link = open_file_within_root(root.join("final.dcm"), &root)
            .expect_err("final symlink must be rejected");
        assert_eq!(final_link.kind(), std::io::ErrorKind::Other);

        remove_tree(&root);
        remove_tree(&outside);
    }

    #[test]
    fn rejects_root_and_directory_as_files() {
        let root = test_root("kind");
        let directory = root.join("series");
        fs::create_dir_all(&directory).expect("test directory creation must succeed");

        let root_error = open_file_within_root(&root, &root).expect_err("root must not be a file");
        let directory_error =
            open_file_within_root(&directory, &root).expect_err("directory must not be a file");
        assert_eq!(root_error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(matches!(
            directory_error.kind(),
            std::io::ErrorKind::InvalidInput
                | std::io::ErrorKind::NotADirectory
                | std::io::ErrorKind::PermissionDenied
        ));

        remove_tree(&root);
    }
}
