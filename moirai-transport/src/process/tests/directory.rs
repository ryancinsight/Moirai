//! Actual child directory and parent isolation evidence.

use std::{
    io::BufReader,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use super::{fixture, line_until};
use crate::process::{
    ProcessDropPolicy, ProcessError, ProcessOperation, ProcessOutcome, ProcessSupervisor,
};

struct Directory(PathBuf);

impl Directory {
    fn create() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "moirai-child-cwd-{}-{id}-Zoë-李-🦀",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("exclusive Unicode directory");
        Self(path.canonicalize().expect("absolute directory"))
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("owned fixture cleanup");
    }
}

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn owned_unicode_directory_changes_only_the_actual_child() {
    let _trace = trace();
    let _test = tracing::info_span!("process_child_directory").entered();
    let (directory, parent) = tracing::info_span!("arrange").in_scope(|| {
        (
            Directory::create(),
            std::env::current_dir().expect("parent cwd"),
        )
    });
    let (actual, status) = tracing::info_span!("act").in_scope(|| {
        let mut child = ProcessSupervisor::new()
            .spawn(
                fixture("cwd").working_directory(&directory.0),
                ProcessDropPolicy::TerminateOnDrop,
            )
            .expect("real child with owned cwd");
        let mut output = BufReader::new(child.take_stdout().expect("child stdout"));
        let actual = line_until(&mut output, "CWD:");
        let status = child
            .wait_timeout(Duration::from_secs(5))
            .expect("finite wait")
            .expect("child exit");
        (actual, status)
    });
    tracing::info_span!("assert").in_scope(|| {
        // Windows current_dir may report the plain spelling of a canonical
        // verbatim directory; comparing the actual filesystem identity avoids
        // treating that platform spelling as a different directory.
        let path = actual
            .trim_end()
            .strip_prefix("CWD:")
            .expect("child cwd marker");
        assert_eq!(
            PathBuf::from(path)
                .canonicalize()
                .expect("child directory identity"),
            directory.0
        );
        assert_eq!(
            std::env::current_dir().expect("unchanged parent cwd"),
            parent
        );
        assert_eq!(status.outcome, ProcessOutcome::Succeeded);
        assert_eq!(status.code, Some(0));
    });
}

#[test]
fn invalid_directory_is_rejected_before_a_child_is_created() {
    let _trace = trace();
    let _test = tracing::info_span!("process_invalid_directory").entered();
    let directory = tracing::info_span!("arrange").in_scope(Directory::create);
    let file = directory.0.join("file");
    std::fs::write(&file, b"real file").expect("non-directory fixture");
    for path in [
        PathBuf::new(),
        PathBuf::from("relative"),
        file,
        directory.0.join("bad\0path"),
    ] {
        let actual = tracing::info_span!("act").in_scope(|| {
            ProcessSupervisor::new().spawn(
                fixture("cwd").working_directory(path),
                ProcessDropPolicy::TerminateOnDrop,
            )
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                actual.expect_err("invalid cwd"),
                ProcessError::InvalidSpecification
            )
        });
    }
    let missing = directory.0.join("missing");
    let expected = std::fs::metadata(&missing)
        .expect_err("missing directory")
        .raw_os_error();
    let actual = tracing::info_span!("act").in_scope(|| {
        ProcessSupervisor::new().spawn(
            fixture("cwd").working_directory(missing),
            ProcessDropPolicy::TerminateOnDrop,
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            actual.expect_err("missing cwd"),
            ProcessError::OperatingSystem {
                operation: ProcessOperation::Spawn,
                code: expected
            }
        );
    });
}
