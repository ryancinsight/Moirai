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
mod unix;

#[cfg(windows)]
mod windows;

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

        // `O_NOFOLLOW` refuses a link with `ELOOP`. std's kind for it
        // (`FilesystemLoop`) is not stable, so the refusal is read from the OS
        // code; it is the kernel's, and the same for a link in the middle of
        // the path and one at its end.
        let intermediate = open_file_within_root(root.join("nested/slice.dcm"), &root)
            .expect_err("intermediate symlink must be rejected");
        assert_eq!(intermediate.raw_os_error(), Some(libc::ELOOP));
        let final_link = open_file_within_root(root.join("final.dcm"), &root)
            .expect_err("final symlink must be rejected");
        assert_eq!(final_link.raw_os_error(), Some(libc::ELOOP));

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
