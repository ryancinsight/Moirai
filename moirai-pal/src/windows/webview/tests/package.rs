//! Owned fixture files and profiles, retained while a native client is uncollected.

use super::WebViewConfig;
use std::{
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

pub(in crate::windows::webview) struct TestPackage {
    pub(in crate::windows::webview) directory: PathBuf,
    pub(in crate::windows::webview) entry: PathBuf,
    pub(in crate::windows::webview) profile: PathBuf,
    cleanup_permitted: bool,
}

impl TestPackage {
    pub(in crate::windows::webview) fn create() -> Self {
        Self::create_with_script(
            br#"<!doctype html><meta charset="utf-8"><script>
window.chrome.webview.postMessage({"ready":true});
</script>"#,
        )
    }

    pub(in crate::windows::webview) fn create_with_script(script: &[u8]) -> Self {
        static NEXT_PACKAGE: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT_PACKAGE.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("moirai-webview2-{}-{id}", std::process::id()));
        std::fs::create_dir(&directory).expect("create unique package directory");
        let entry = directory.join("index.html");
        std::fs::write(&entry, script).expect("write package entry");
        let profile = directory.with_file_name(format!(
            "moirai-webview2-profile-{}-{id}",
            std::process::id()
        ));
        Self {
            directory,
            entry,
            profile,
            cleanup_permitted: true,
        }
    }

    pub(in crate::windows::webview) fn fresh_config(&self) -> WebViewConfig {
        std::fs::create_dir(&self.profile).expect("create exclusively owned fresh profile");
        WebViewConfig::folder(
            "metis.invalid",
            &self.directory,
            "index.html",
            Duration::from_millis(u64::from(
                crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS,
            )),
        )
        .expect("secure packaged origin")
        .with_user_data_folder(&self.profile)
        .expect("fresh local profile")
    }

    /// Every native owner must finish before the fixture can remove its files.
    /// Disable cleanup before spawning, so unwinding also retains ownership.
    pub(in crate::windows::webview) fn permit_cleanup(&mut self, permitted: bool) {
        self.cleanup_permitted = permitted;
    }

    pub(in crate::windows::webview) fn cleanup(&self) -> io::Result<()> {
        if !self.cleanup_permitted {
            return Err(io::Error::other(format!(
                "Native ownership is unconfirmed; retained package {} and profile {}",
                self.directory.display(),
                self.profile.display(),
            )));
        }
        std::fs::remove_dir_all(&self.directory)?;
        std::fs::remove_dir_all(&self.profile)
    }

    pub(super) fn uri(&self) -> String {
        let entry = self
            .entry
            .canonicalize()
            .expect("temporary package entry is canonical");
        file_uri(&entry)
    }
}

impl Drop for TestPackage {
    fn drop(&mut self) {
        if !self.cleanup_permitted {
            tracing::error!(package = %self.directory.display(), profile = %self.profile.display(), "unconfirmed native ownership retains fixture directories");
            return;
        }
        for folder in [&self.directory, &self.profile] {
            if let Err(error) = std::fs::remove_dir_all(folder)
                && error.kind() != io::ErrorKind::NotFound
            {
                tracing::error!(path = %folder.display(), error = %error, "WebView2 fixture directory cleanup failed");
            }
        }
    }
}

pub(super) fn file_uri(path: &Path) -> String {
    let path = path
        .to_str()
        .expect("temporary package path is valid UTF-8")
        .replace('\\', "/");
    let path = path.strip_prefix("//?/").unwrap_or(&path);
    let mut uri = String::from("file:///");
    for byte in path.bytes() {
        if matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' | b':')
        {
            uri.push(char::from(byte));
        } else {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            uri.push('%');
            uri.push(char::from(HEX[usize::from(byte >> 4)]));
            uri.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
    }
    uri
}

#[cfg(test)]
mod tests {
    use super::TestPackage;
    use crate::windows::webview::tests::startup::tracing;

    #[test]
    fn deferred_cleanup_preserves_files_during_explicit_and_drop_cleanup() {
        let (_trace, _subscriber) = tracing();
        let _test = tracing::info_span!("fixture_deferred_cleanup").entered();
        let (mut package, directory, profile, entry) =
            tracing::info_span!("arrange").in_scope(|| {
                let package = TestPackage::create_with_script("<p>患者-🦀</p>".as_bytes());
                package.fresh_config();
                let paths = (
                    package.directory.clone(),
                    package.profile.clone(),
                    package.entry.clone(),
                );
                (package, paths.0, paths.1, paths.2)
            });
        let error = tracing::info_span!("act").in_scope(|| {
            package.permit_cleanup(false);
            let error = package.cleanup().expect_err("deferred client collection");
            drop(package);
            error
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), std::io::ErrorKind::Other);
            assert!(error.to_string().contains("ownership is unconfirmed"));
            assert_eq!(
                std::fs::read(&entry).expect("retained actual entry"),
                "<p>患者-🦀</p>".as_bytes()
            );
            assert!(directory.is_dir() && profile.is_dir());
        });
        // This policy test creates no client. Remove its retained real files
        // only after observing the explicit and Drop cleanup contracts.
        std::fs::remove_dir_all(directory).expect("owned policy-test package");
        std::fs::remove_dir_all(profile).expect("owned policy-test profile");
    }

    #[test]
    fn admitted_cleanup_removes_both_actual_owned_directories() {
        let (_trace, _subscriber) = tracing();
        let _test = tracing::info_span!("fixture_admitted_cleanup").entered();
        let mut package = tracing::info_span!("arrange").in_scope(|| {
            let mut package = TestPackage::create();
            package.fresh_config();
            package.permit_cleanup(false);
            package
        });
        tracing::info_span!("act").in_scope(|| {
            package.permit_cleanup(true);
            package.cleanup().expect("admitted file cleanup");
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(!package.directory.exists() && !package.profile.exists());
        });
    }
}
