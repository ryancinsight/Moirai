use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn capture() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "moirai-webview-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("unique test directory");
        fs::write(
            path.join("index.html"),
            b"<!doctype html><title>resource oracle</title>",
        )
        .expect("real entry");
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("owned test directory cleanup");
    }
}

#[test]
fn reads_exact_bytes_and_mime_from_packaged_files() {
    let _trace = capture();
    let _test = tracing::info_span!("resource_bytes_and_mime").entered();
    let (dir, root) = tracing::info_span!("arrange").in_scope(|| {
        let dir = Directory::new();
        let root = ResourceRoot::new(&dir.0).expect("resource root");
        (dir, root)
    });
    for (name, bytes, mime) in [
        (
            "module.wasm",
            b"\0asm\x01\0\0\0".as_slice(),
            "application/wasm",
        ),
        (
            "script.js",
            b"export const value = 13;".as_slice(),
            "text/javascript",
        ),
        ("a b.css", b"body { color: blue }".as_slice(), "text/css"),
    ] {
        fs::write(dir.0.join(name), bytes).expect("resource bytes");
        let uri = format!("metis://app/{}?revision=1", name.replace(' ', "%20"));
        let result =
            tracing::info_span!("act", %uri).in_scope(|| root.read(&uri).expect("packaged file"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(result, (bytes.to_vec(), mime));
        });
    }
}

#[test]
fn rejects_other_origins_and_decoded_traversal() {
    let _trace = capture();
    let _test = tracing::info_span!("resource_authority").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let dir = Directory::new();
    let root = ResourceRoot::new(&dir.0).expect("resource root");
    drop(arrange);
    for uri in [
        "https://app/index.html",
        "metis://app.evil/index.html",
        "metis://app@evil/index.html",
        "metis://app/../index.html",
        "metis://app/%2e%2e/index.html",
        "metis://app/%2Fetc/passwd",
        "metis://app/%5Cindex.html",
    ] {
        let error = tracing::info_span!("act", %uri).in_scope(|| root.read(uri).expect_err(uri));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        });
    }
    for uri in [
        "metis://app/%",
        "metis://app/%0g",
        "metis://app/%ff",
        "metis://app/%00index.html",
    ] {
        let error = tracing::info_span!("act", %uri).in_scope(|| root.read(uri).expect_err(uri));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        });
    }
    assert!(ResourceRoot::permits("metis://app/index.html"));
    assert!(!ResourceRoot::permits("metis://app.evil/index.html"));
}

#[test]
fn rejects_missing_and_oversized_files() {
    let _trace = capture();
    let _test = tracing::info_span!("resource_negative_bounds").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let dir = Directory::new();
    let root = ResourceRoot::new(&dir.0).expect("resource root");
    fs::File::create(dir.0.join("huge.wasm"))
        .expect("sparse resource")
        .set_len(MAX_RESOURCE_BYTES + 1)
        .expect("sparse bound violation");
    drop(arrange);
    for (uri, kind) in [
        ("metis://app/missing.html", io::ErrorKind::NotFound),
        ("metis://app/huge.wasm", io::ErrorKind::InvalidData),
    ] {
        let error = tracing::info_span!("act", %uri)
            .in_scope(|| root.read(uri).expect_err("rejected resource"));
        tracing::info_span!("assert").in_scope(|| assert_eq!(error.kind(), kind));
    }
}

#[test]
fn admits_the_exact_byte_budget_with_input_sensitive_content() {
    use std::io::Write;
    let _trace = capture();
    let _test = tracing::info_span!("resource_inclusive_bound").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let dir = Directory::new();
    let root = ResourceRoot::new(&dir.0).expect("resource root");
    // A repeating complete byte alphabet has an analytical oracle at every
    // offset: byte(i) = i mod 256, including both ends of the resource budget.
    let alphabet: Vec<u8> = (u8::MIN..=u8::MAX).collect();
    let mut file = std::io::BufWriter::new(
        fs::File::create(dir.0.join("boundary.bin")).expect("boundary file"),
    );
    for _ in 0..MAX_RESOURCE_BYTES / u64::try_from(alphabet.len()).expect("alphabet size") {
        file.write_all(&alphabet).expect("boundary content");
    }
    file.flush().expect("boundary content flush");
    drop(file);
    drop(arrange);
    let (bytes, mime) = tracing::info_span!("act").in_scope(|| {
        root.read("metis://app/boundary.bin")
            .expect("inclusive budget")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            u64::try_from(bytes.len()).expect("byte count"),
            MAX_RESOURCE_BYTES
        );
        assert_eq!(mime, "application/octet-stream");
        assert!(
            bytes
                .chunks_exact(alphabet.len())
                .all(|chunk| chunk == alphabet)
        );
    });
}

#[cfg(unix)]
#[test]
fn rejects_special_files_before_opening_them() {
    let _trace = capture();
    let _test = tracing::info_span!("resource_regular_file_contract").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let dir = Directory::new();
    let root = ResourceRoot::new(&dir.0).expect("resource root");
    let _socket = std::os::unix::net::UnixListener::bind(dir.0.join("socket"))
        .expect("real special-file fixture");
    let status = std::process::Command::new("mkfifo")
        .arg(dir.0.join("pipe"))
        .status()
        .expect("create real FIFO fixture");
    assert_eq!(status.code(), Some(0));
    use std::os::unix::fs::FileTypeExt;
    assert!(
        fs::metadata(dir.0.join("pipe"))
            .expect("FIFO metadata")
            .file_type()
            .is_fifo()
    );
    drop(arrange);
    for name in ["socket", "pipe"] {
        let error = tracing::info_span!("act", name).in_scope(|| {
            root.read(&format!("metis://app/{name}"))
                .expect_err("regular files only")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert_eq!(error.to_string(), "resource is not a regular file");
        });
    }
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_outside_the_resource_root() {
    let _trace = capture();
    let _test = tracing::info_span!("resource_symlink_containment").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let dir = Directory::new();
    let outside = Directory::new();
    let root = ResourceRoot::new(&dir.0).expect("resource root");
    std::os::unix::fs::symlink(outside.0.join("index.html"), dir.0.join("escape.html"))
        .expect("real symlink");
    drop(arrange);
    let error = tracing::info_span!("act").in_scope(|| {
        root.read("metis://app/escape.html")
            .expect_err("symlink containment")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    });
}
