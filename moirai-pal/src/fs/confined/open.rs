//! Confined opening entry point and lexical path validation.
//! The platform backends live behind it in unix and windows.

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
            super::unix::open(root, &relative)
        }

        #[cfg(windows)]
        {
            super::windows::open(root, &relative)
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
