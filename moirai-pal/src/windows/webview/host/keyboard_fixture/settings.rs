//! Fixture-only global setting, applied before the first configured navigation.
//!
//! WebView2 documents that a changed browser-accelerator setting takes effect
//! after navigation. Controller creation can already have loaded about:blank;
//! this hook precedes navigation to the real Rust payload. It is compiled only
//! into tests and makes no SDK calls when selection is absent.

mod selection;

use super::policy::{Failure, checked};
use serde::Serialize;
use std::io;
use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2, ICoreWebView2Settings, ICoreWebView2Settings3,
};
use windows::core::{BOOL, Interface};

#[derive(Debug, Serialize)]
pub(in crate::windows::webview::host) struct Attempt {
    pub(super) requested: bool,
    before: Option<bool>,
    setter_succeeded: Option<bool>,
    after: Option<bool>,
    after_navigation: Option<bool>,
    failure: Option<Failure>,
}

pub(super) fn selected() -> io::Result<Option<bool>> {
    selection::selection(
        std::env::var_os("MOIRAI_WEBVIEW_KEYBOARD_SETTINGS").as_deref(),
        std::env::var("MOIRAI_WEBVIEW_KEYBOARD_POLICY").as_deref() == Ok("default"),
        std::env::var("MOIRAI_WEBVIEW_KEYBOARD_OBSERVER").as_deref() != Ok("none"),
    )
}

pub(in crate::windows::webview::host) fn before_navigation(
    settings: &ICoreWebView2Settings,
    configured_uri: &str,
) -> io::Result<Option<Attempt>> {
    let Some(requested) = selected()? else {
        return Ok(None);
    };
    if configured_uri != super::session::URI {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "global keyboard setting requires the real keyboard fixture URI",
        ));
    }
    let mut attempt = Attempt {
        requested,
        before: None,
        setter_succeeded: None,
        after: None,
        after_navigation: None,
        failure: None,
    };
    let result = attempt.apply(settings);
    attempt.failure = result.as_ref().err().cloned();
    tracing::info!(event="keyboard_global_setting_before_navigation", observation=?attempt,
        "actual fixture setting before configured navigation");
    result.map_err(io::Error::other)?;
    Ok(Some(attempt))
}

impl Attempt {
    fn apply(&mut self, settings: &ICoreWebView2Settings) -> Result<(), Failure> {
        let extended = checked(
            "QueryInterface Settings3 before navigation",
            settings.cast::<ICoreWebView2Settings3>(),
        )?;
        let mut enabled = BOOL(0);
        // SAFETY: the live settings interface and writable output outlive each
        // synchronous SDK call; no callback or interface escapes this method.
        unsafe {
            checked(
                "AreBrowserAcceleratorKeysEnabled before",
                extended.AreBrowserAcceleratorKeysEnabled(&mut enabled),
            )?;
            self.before = Some(enabled.as_bool());
            let result = extended.SetAreBrowserAcceleratorKeysEnabled(self.requested);
            self.setter_succeeded = Some(result.is_ok());
            checked("SetAreBrowserAcceleratorKeysEnabled", result)?;
            checked(
                "AreBrowserAcceleratorKeysEnabled after",
                extended.AreBrowserAcceleratorKeysEnabled(&mut enabled),
            )?;
            self.after = Some(enabled.as_bool());
        }
        Ok(())
    }

    pub(super) fn verify_after_navigation(&mut self, webview: &ICoreWebView2) -> io::Result<()> {
        let result = self.read_after_navigation(webview);
        self.failure = result.as_ref().err().cloned();
        result.map_err(io::Error::other)?;
        if self.before != Some(true)
            || self.setter_succeeded != Some(true)
            || self.after != Some(self.requested)
            || self.after_navigation != Some(self.requested)
        {
            return Err(io::Error::other(
                "actual global keyboard setting differs from the fresh-profile control",
            ));
        }
        Ok(())
    }

    fn read_after_navigation(&mut self, webview: &ICoreWebView2) -> Result<(), Failure> {
        // SAFETY: the fixture's still-owned WebView transfers a live settings
        // interface and writes a BOOL valid for this synchronous readback.
        unsafe {
            let settings = checked("Settings after navigation", webview.Settings())?;
            let extended = checked(
                "QueryInterface Settings3 after navigation",
                settings.cast::<ICoreWebView2Settings3>(),
            )?;
            let mut enabled = BOOL(0);
            checked(
                "AreBrowserAcceleratorKeysEnabled after navigation",
                extended.AreBrowserAcceleratorKeysEnabled(&mut enabled),
            )?;
            self.after_navigation = Some(enabled.as_bool());
        }
        Ok(())
    }
}
