//! Actual filesystem admission and revalidation, without COM substitution.

use std::{io::ErrorKind, path::Path};

use super::validate;
use crate::windows::webview::tests::TestPackage;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn fresh_profile_uses_the_actual_empty_local_directory() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_fresh_profile").entered();
    let package = tracing::info_span!("arrange").in_scope(TestPackage::create);
    std::fs::create_dir(&package.profile).expect("exclusive profile creation");
    let actual = tracing::info_span!("act").in_scope(|| validate(&package.profile));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(actual.expect("empty local directory"), package.profile);
        assert_eq!(
            std::fs::read_dir(&package.profile)
                .expect("profile directory")
                .count(),
            0
        );
    });
}

#[test]
fn profile_rejects_remote_relative_file_and_nonempty_paths() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_profile_rejections").entered();
    let package = tracing::info_span!("arrange").in_scope(TestPackage::create);
    std::fs::create_dir(&package.profile).expect("exclusive profile creation");
    std::fs::write(package.profile.join("existing"), b"retained user data").expect("existing data");
    for (path, expected) in [
        (
            Path::new("relative"),
            "WebView2 profile requires an absolute local directory",
        ),
        (
            Path::new(r"\\server\share\profile"),
            "WebView2 profile requires a local drive directory",
        ),
        (
            package.entry.as_path(),
            "WebView2 profile target is not a directory",
        ),
        (
            package.profile.as_path(),
            "WebView2 profile directory must be empty before creation",
        ),
    ] {
        let actual = tracing::info_span!("act").in_scope(|| validate(path));
        tracing::info_span!("assert").in_scope(|| {
            let error = actual.expect_err("invalid profile");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert_eq!(error.to_string(), expected);
        });
    }
    assert_eq!(
        std::fs::read(package.profile.join("existing")).expect("data retained"),
        b"retained user data"
    );
}

#[test]
fn config_revalidation_detects_intervening_profile_data() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_profile_revalidation").entered();
    let (package, config) = tracing::info_span!("arrange").in_scope(|| {
        let package = TestPackage::create();
        let config = package.fresh_config();
        std::fs::write(package.profile.join("appeared"), b"changed before COM")
            .expect("intervening data");
        (package, config)
    });
    let actual = tracing::info_span!("act").in_scope(|| config.validate_directories());
    tracing::info_span!("assert").in_scope(|| {
        let error = actual.expect_err("changed profile");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "WebView2 profile directory must be empty before creation"
        );
        assert_eq!(
            std::fs::read(package.profile.join("appeared")).expect("data retained"),
            b"changed before COM"
        );
    });
}

#[test]
fn actual_directory_junction_is_rejected_before_profile_creation() {
    use std::os::windows::fs::MetadataExt;
    use windows::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

    let _trace = trace();
    let _test = tracing::info_span!("webview_profile_reparse_rejection").entered();
    let package = tracing::info_span!("arrange").in_scope(TestPackage::create);
    std::fs::create_dir(&package.profile).expect("exclusive profile creation");
    let junction = package.directory.join("profile-link");
    // NTFS junction creation requires no symlink privilege and exercises the
    // actual reparse metadata consumed by the production path validator.
    let result = std::process::Command::new("cmd.exe")
        .args(["/c", "mklink", "/J"])
        .arg(&junction)
        .arg(&package.profile)
        .output()
        .expect("native NTFS junction creation");
    assert!(
        result.status.success(),
        "junction creation failed: {result:?}"
    );
    let actual = tracing::info_span!("act").in_scope(|| validate(&junction));
    tracing::info_span!("assert").in_scope(|| {
        let attributes = std::fs::symlink_metadata(&junction)
            .expect("junction metadata")
            .file_attributes();
        assert_ne!(attributes & FILE_ATTRIBUTE_REPARSE_POINT.0, 0);
        let error = actual.expect_err("reparse profile path");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "WebView2 profile path contains a reparse point"
        );
        assert_eq!(
            std::fs::read_dir(&package.profile)
                .expect("profile remains empty")
                .count(),
            0
        );
    });
}

#[test]
fn mapped_package_profile_overlap_uses_native_case_insensitive_components() {
    use crate::windows::webview::WebViewConfig;
    use std::time::Duration;

    let _trace = trace();
    let _test = tracing::info_span!("webview_profile_package_overlap").entered();
    let (package, nested) = tracing::info_span!("arrange").in_scope(|| {
        let package = TestPackage::create();
        let nested = package.directory.join("fresh-profile");
        std::fs::create_dir(&nested).expect("empty directory within mapped package");
        (package, nested)
    });
    let actual = tracing::info_span!("act").in_scope(|| {
        WebViewConfig::folder(
            "metis.invalid",
            &package.directory,
            "index.html",
            Duration::from_secs(1),
        )
        .expect("actual package folder")
        .with_user_data_folder(nested.to_str().expect("fixture path").to_lowercase())
    });
    tracing::info_span!("assert").in_scope(|| {
        let error = actual.expect_err("mapped package cannot serve profile resources");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "WebView2 profile and mapped package folders must not overlap"
        );
        assert_eq!(
            std::fs::read_dir(&nested)
                .expect("profile remains empty")
                .count(),
            0
        );
    });
}

#[test]
fn profile_admission_rejects_control_surrogate_and_excessive_native_units() {
    use super::MAX_PROFILE_PATH_UNITS;
    use std::{ffi::OsString, os::windows::ffi::OsStringExt, path::PathBuf};

    let _trace = trace();
    let _test = tracing::info_span!("webview_profile_native_representations").entered();
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        [
            PathBuf::from("C:\\bad\0path"),
            PathBuf::from(OsString::from_wide(&[0x0043, 0x003a, 0x005c, 0xd800])),
            PathBuf::from(format!("C:\\{}", "a".repeat(MAX_PROFILE_PATH_UNITS - 2))),
        ]
    });
    for path in inputs {
        let actual = tracing::info_span!("act").in_scope(|| validate(&path));
        tracing::info_span!("assert").in_scope(|| {
            let error = actual.expect_err("invalid native path representation");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "WebView2 profile path exceeds its Unicode native representation bound"
            );
        });
    }
}
