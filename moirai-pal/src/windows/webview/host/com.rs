//! WebView2 COM apartment and asynchronous creation.

use std::{io, path::Path, sync::mpsc, time::Duration};

use webview2_com::Microsoft::Web::WebView2::Win32::{
    CreateCoreWebView2EnvironmentWithOptions, ICoreWebView2Controller, ICoreWebView2Environment,
    ICoreWebView2Environment7,
};
use webview2_com::{
    CreateCoreWebView2ControllerCompletedHandler, CreateCoreWebView2EnvironmentCompletedHandler,
};
use windows::{
    Win32::{
        Foundation::{E_POINTER, HWND},
        System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx, CoUninitialize},
    },
    core::{Error, HSTRING, Interface, PCWSTR, PWSTR},
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

pub(super) fn create_environment(
    wait: Duration,
    profile: Option<&Path>,
) -> io::Result<ICoreWebView2Environment> {
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
    let profile_string = profile.map(|path| HSTRING::from(path.as_os_str()));
    let profile_path = profile_string
        .as_ref()
        .map_or(PCWSTR::null(), |path| PCWSTR(path.as_ptr()));
    // SAFETY: the SDK copies these optional UTF16 arguments during this call;
    // the owned profile string and callback outlive the call and finite wait.
    unsafe {
        CreateCoreWebView2EnvironmentWithOptions(PCWSTR::null(), profile_path, None, &handler)
    }
    .map_err(windows_error)?;
    let environment = wait_for(receiver, wait)?;
    if let Some(expected) = profile {
        let properties = environment
            .cast::<ICoreWebView2Environment7>()
            .map_err(|error| {
                io::Error::other(format!("WebView2 effective-profile interface: {error}"))
            })?;
        let mut raw_path = PWSTR::null();
        // SAFETY: this exclusive getter output transfers one CoTaskMem buffer.
        unsafe { properties.UserDataFolder(&mut raw_path) }.map_err(windows_error)?;
        // SAFETY: nothing else uses or frees the transferred getter buffer.
        let actual = unsafe { super::text::read_task_mem_profile(raw_path) }.map_err(|error| {
            io::Error::other(format!("WebView2 actual user-data directory: {error}"))
        })?;
        super::super::config::profile::verify_effective(expected, Path::new(&actual))?;
    }
    Ok(environment)
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
