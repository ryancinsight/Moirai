//! Actual fresh-profile browser shutdown before caller-owned directory cleanup.

mod observer;

use std::io;

use super::{
    callbacks::{
        remove_message, remove_navigation_completed, remove_navigation_starting, remove_new_window,
        remove_permission_requested,
    },
    error::windows_error,
    view::WebViewHost,
};

impl WebViewHost {
    /// Closes callbacks, the controller and the parent window synchronously.
    ///
    /// All cleanup steps are attempted. The first native error is returned
    /// after the remaining handles have been released.
    /// An explicitly configured fresh profile also waits for its actual
    /// `BrowserProcessExited` event before releasing the environment, proving
    /// the runtime has released the caller's user-data directory. Shared default
    /// profiles retain their existing controller-only close behavior.
    ///
    /// # Errors
    /// Returns the first callback-removal, controller-close or window-close
    /// error observed during teardown, or owned-profile browser-exit observation
    /// failure. A failed close remains failed on subsequent calls.
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return match &self.close_error {
                Some((kind, context)) => Err(io::Error::new(*kind, context.clone())),
                None => Ok(()),
            };
        }
        let mut first_error = None;
        let observer = if self.config.user_data_folder().is_some() {
            match self.webview.as_ref().zip(self.environment.as_ref()) {
                Some((webview, environment)) => {
                    match observer::BrowserExit::register(webview, environment) {
                        Ok(observer) => Some(observer),
                        Err(error) => {
                            first_error = Some(error);
                            None
                        }
                    }
                }
                None => {
                    first_error = Some(io::Error::new(
                        io::ErrorKind::NotConnected,
                        "WebView2 owned profile has no live browser to observe",
                    ));
                    None
                }
            }
        } else {
            None
        };
        if let Some(webview) = self.webview.as_ref() {
            remove_navigation_starting(
                webview,
                &mut self.callbacks.navigation_starting,
                &mut first_error,
            );
            remove_navigation_completed(
                webview,
                &mut self.callbacks.navigation_completed,
                &mut first_error,
            );
            remove_new_window(webview, &mut self.callbacks.new_window, &mut first_error);
            remove_permission_requested(
                webview,
                &mut self.callbacks.permission_requested,
                &mut first_error,
            );
            remove_message(webview, &mut self.callbacks.message, &mut first_error);
        }
        if let Some(controller) = self.controller.take()
            && let Err(error) = unsafe { controller.Close() }.map_err(windows_error)
        {
            first_error.get_or_insert(error);
        }
        self.webview.take();
        if let Some(observer) = observer
            && let Err(error) = observer.wait(self.config.wait())
        {
            first_error.get_or_insert(error);
        }
        self.environment.take();
        if let Err(error) = self.window.close() {
            first_error.get_or_insert(error);
        }
        self.closed = true;
        self.close_error = first_error
            .as_ref()
            .map(|error| (error.kind(), error.to_string()));
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}
