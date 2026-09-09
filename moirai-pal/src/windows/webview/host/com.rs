//! WebView2 COM apartment and asynchronous creation.

use std::{io, sync::mpsc, time::Duration};

use webview2_com::Microsoft::Web::WebView2::Win32::{
    CreateCoreWebView2Environment, ICoreWebView2Controller, ICoreWebView2Environment,
};
use webview2_com::{
    CreateCoreWebView2ControllerCompletedHandler, CreateCoreWebView2EnvironmentCompletedHandler,
};
use windows::{
    Win32::{
        Foundation::{E_POINTER, HWND},
        System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx, CoUninitialize},
    },
    core::Error,
};

use super::super::pump::wait_for;
use super::error::{callback_error, windows_error};

pub(super) struct ComApartment {
    pub(super) initialized: bool,
}

impl ComApartment {
    pub(super) fn initialize() -> io::Result<Self> {
        // SAFETY: COM is initialized once on the creating thread and the
        // matching uninitialization is performed by this RAII guard.
        let result = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        if result.is_ok() {
            Ok(Self { initialized: true })
        } else {
            Err(windows_error(result.into()))
        }
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.initialized {
            // SAFETY: this guard is dropped on the same thread that initialized
            // the apartment, after all WebView2 interfaces have been released.
            unsafe { CoUninitialize() };
        }
    }
}

pub(super) fn create_environment(wait: Duration) -> io::Result<ICoreWebView2Environment> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let handler = CreateCoreWebView2EnvironmentCompletedHandler::create(Box::new(
        move |error_code, environment| {
            let result =
                error_code.and_then(|()| environment.ok_or_else(|| Error::from(E_POINTER)));
            sender
                .send(result)
                .map_err(|_| callback_error("WebView2 environment waiter was dropped"))
        },
    ));
    unsafe { CreateCoreWebView2Environment(&handler) }.map_err(windows_error)?;
    wait_for(receiver, wait)
}

pub(super) fn create_controller(
    environment: &ICoreWebView2Environment,
    parent: HWND,
    wait: Duration,
) -> io::Result<ICoreWebView2Controller> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let environment = environment.clone();
    let handler = CreateCoreWebView2ControllerCompletedHandler::create(Box::new(
        move |error_code, controller| {
            let result = error_code.and_then(|()| controller.ok_or_else(|| Error::from(E_POINTER)));
            sender
                .send(result)
                .map_err(|_| callback_error("WebView2 controller waiter was dropped"))
        },
    ));
    unsafe {
        environment
            .CreateCoreWebView2Controller(parent, &handler)
            .map_err(windows_error)?;
    }
    wait_for(receiver, wait)
}
