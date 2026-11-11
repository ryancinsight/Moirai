use super::open::open_file_within_root;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_root(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "moirai_pal_confined_{name}_{}_{}",
        std::process::id(),
        nonce
    ));
    // Nonce-named yet created without `create_new`: pid reuse plus a stale
    // tree from a crashed run can collide, so remove any leftover first —
    // best-effort, since a passed run must not fail on a contended temp dir.
    let _ = fs::remove_dir_all(&root);
    root
}

/// RAII owner for a `test_root` tree: the tree is removed when the guard
/// drops, so a test that fails or panics mid-body cannot leak its temp dir.
struct TestRoot {
    path: PathBuf,
}

impl TestRoot {
    fn create(name: &str) -> Self {
        let path = test_root(name);
        fs::create_dir_all(&path).expect("test tree creation must succeed");
        Self { path }
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn opens_nested_regular_file_and_returns_source_handle() {
    let root = TestRoot::create("nested");
    let directory = root.path.join("series");
    let path = directory.join("slice.dcm");
    fs::create_dir_all(&directory).expect("test directory creation must succeed");
    fs::write(&path, b"dicom-bytes").expect("test file creation must succeed");

    let mut file = open_file_within_root(&path, &root.path).expect("confined open must succeed");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .expect("confined handle read must succeed");
    assert_eq!(bytes, b"dicom-bytes");
}

#[test]
fn rejects_parent_and_absolute_escape_paths() {
    let root = TestRoot::create("escape");
    fs::write(root.path.join("slice.dcm"), b"safe").expect("test file creation must succeed");

    let parent = open_file_within_root(root.path.join("..").join("outside.dcm"), &root.path)
        .expect_err("parent traversal must be rejected");
    assert_eq!(parent.kind(), std::io::ErrorKind::InvalidInput);

    let outside = test_root("outside").join("slice.dcm");
    let absolute = open_file_within_root(&outside, &root.path)
        .expect_err("absolute path outside root must be rejected");
    assert_eq!(absolute.kind(), std::io::ErrorKind::PermissionDenied);
}

#[cfg(unix)]
#[test]
fn rejects_intermediate_and_final_symlinks() {
    use std::os::unix::fs::symlink;

    let root = TestRoot::create("links");
    let outside = TestRoot::create("link-target");
    fs::write(outside.path.join("slice.dcm"), b"outside")
        .expect("target file creation must succeed");
    fs::write(root.path.join("real.dcm"), b"inside").expect("root file creation must succeed");
    symlink(&outside.path, root.path.join("nested"))
        .expect("intermediate link creation must succeed");
    symlink(outside.path.join("slice.dcm"), root.path.join("final.dcm"))
        .expect("final link creation must succeed");

    // `O_NOFOLLOW` refuses a link with `ELOOP`. std's kind for it
    // (`FilesystemLoop`) is not stable, so the refusal is read from the OS
    // code; it is the kernel's, and the same for a link in the middle of
    // the path and one at its end.
    let intermediate = open_file_within_root(root.path.join("nested/slice.dcm"), &root.path)
        .expect_err("intermediate symlink must be rejected");
    assert_eq!(intermediate.raw_os_error(), Some(libc::ELOOP));
    let final_link = open_file_within_root(root.path.join("final.dcm"), &root.path)
        .expect_err("final symlink must be rejected");
    assert_eq!(final_link.raw_os_error(), Some(libc::ELOOP));
}

#[test]
fn rejects_root_and_directory_as_files() {
    let root = TestRoot::create("kind");
    let directory = root.path.join("series");
    fs::create_dir_all(&directory).expect("test directory creation must succeed");

    let root_error =
        open_file_within_root(&root.path, &root.path).expect_err("root must not be a file");
    let directory_error =
        open_file_within_root(&directory, &root.path).expect_err("directory must not be a file");
    assert_eq!(root_error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(matches!(
        directory_error.kind(),
        std::io::ErrorKind::InvalidInput
            | std::io::ErrorKind::NotADirectory
            | std::io::ErrorKind::PermissionDenied
    ));
}

#[cfg(unix)]
#[test]
fn a_fifo_is_refused_without_waiting_for_a_writer() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let root = TestRoot::create("fifo");
    fs::create_dir_all(&root.path).expect("test directory creation must succeed");
    let fifo = root.path.join("pipe");
    let name = CString::new(fifo.as_os_str().as_bytes()).expect("the path has no NUL");
    // SAFETY: `name` is a NUL-terminated path in a directory this test owns.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);

    // No writer exists. A blocking open would never return; the non-blocking
    // open returns and the file-type check refuses the FIFO.
    let error = open_file_within_root(&fifo, &root.path).expect_err("a FIFO is not a regular file");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}
