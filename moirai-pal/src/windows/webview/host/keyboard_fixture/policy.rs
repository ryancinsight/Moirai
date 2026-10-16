//! A fixture-owned synchronous intervention, separate from passive diagnostics.

use serde::Serialize;
use std::{cell::RefCell, io, rc::Rc};
use webview2_com::{
    AcceleratorKeyPressedEventHandler,
    Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_KEY_EVENT_KIND, ICoreWebView2AcceleratorKeyPressedEventArgs,
        ICoreWebView2AcceleratorKeyPressedEventArgs2,
        ICoreWebView2AcceleratorKeyPressedEventHandler, ICoreWebView2Controller,
    },
};
use windows::{
    Win32::UI::Input::KeyboardAndMouse::{
        GetKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    },
    core::{BOOL, Interface},
};

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub(super) enum Policy {
    Default,
    EnableS,
    SkipS,
}

impl Policy {
    fn override_value(self) -> Option<bool> {
        match self {
            Self::Default => None,
            Self::EnableS => Some(true),
            Self::SkipS => Some(false),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Failure {
    operation: &'static str,
    hresult: Option<i32>,
    #[serde(skip)]
    native: Option<windows::core::Error>,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{}: {:?}", self.operation, self.hresult)
    }
}

impl std::error::Error for Failure {
    // Erasure is the cold standard error-source ABI for heterogeneous SDK errors.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.native
            .as_ref()
            .map(|error| error as &(dyn std::error::Error + 'static))
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct Attempt {
    pub(super) kind: Option<i32>,
    pub(super) key: Option<u32>,
    pub(super) lparam: Option<i32>,
    pub(super) held: Option<[i16; 5]>,
    pub(super) handled_before: Option<bool>,
    pub(super) browser_before: Option<bool>,
    pub(super) setter_succeeded: Option<bool>,
    pub(super) browser_after: Option<bool>,
    pub(super) handled_after: Option<bool>,
    pub(super) failure: Option<Failure>,
}

impl Attempt {
    fn capture(
        &mut self,
        args: Option<&ICoreWebView2AcceleratorKeyPressedEventArgs>,
        policy: Policy,
    ) -> Result<(), Failure> {
        let args = args.ok_or(Failure {
            operation: "missing SDK arguments",
            hresult: None,
            native: None,
        })?;
        let mut kind = COREWEBVIEW2_KEY_EVENT_KIND(0);
        let mut key = 0;
        let mut lparam = 0;
        let mut handled = BOOL(0);
        // SAFETY: supplied arguments and all getter outputs live through this callback.
        unsafe {
            checked("KeyEventKind", args.KeyEventKind(&mut kind))?;
            self.kind = Some(kind.0);
            checked("VirtualKey", args.VirtualKey(&mut key))?;
            self.key = Some(key);
            checked("KeyEventLParam", args.KeyEventLParam(&mut lparam))?;
            self.lparam = Some(lparam);
            self.held = Some(
                [VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN]
                    .map(|vk| GetKeyState(i32::from(vk.0))),
            );
            checked("Handled before", args.Handled(&mut handled))?;
            self.handled_before = Some(handled.as_bool());
        }
        let extended = checked(
            "QueryInterface Args2",
            args.cast::<ICoreWebView2AcceleratorKeyPressedEventArgs2>(),
        )?;
        let mut enabled = BOOL(0);
        // SAFETY: the queried event interface remains local to this invocation.
        unsafe {
            checked(
                "browser property before",
                extended.IsBrowserAcceleratorKeyEnabled(&mut enabled),
            )?;
            self.browser_before = Some(enabled.as_bool());
            if matches_system(self)
                && let Some(value) = policy.override_value()
            {
                let result = extended.SetIsBrowserAcceleratorKeyEnabled(value);
                self.setter_succeeded = Some(result.is_ok());
                checked("SetIsBrowserAcceleratorKeyEnabled", result)?;
            }
            checked(
                "browser property after",
                extended.IsBrowserAcceleratorKeyEnabled(&mut enabled),
            )?;
            self.browser_after = Some(enabled.as_bool());
            checked("Handled after", args.Handled(&mut handled))?;
            self.handled_after = Some(handled.as_bool());
        }
        Ok(())
    }
}

pub(super) fn checked<T>(
    operation: &'static str,
    result: windows::core::Result<T>,
) -> Result<T, Failure> {
    result.map_err(|error| Failure {
        operation,
        hresult: Some(error.code().0),
        native: Some(error),
    })
}

pub(super) fn matches_system(record: &Attempt) -> bool {
    let (Some(kind), Some(83), Some(bits), Some(held)) =
        (record.kind, record.key, record.lparam, record.held)
    else {
        return false;
    };
    matches!(kind, 0 | 2)
        && bits.cast_unsigned() & (1 << 30) == 0
        && held[0] >= 0
        && held[1] < 0
        && held[2] < 0
        && held[3] >= 0
        && held[4] >= 0
}

pub(super) struct Registration {
    controller: ICoreWebView2Controller,
    _handler: ICoreWebView2AcceleratorKeyPressedEventHandler,
    token: Option<i64>,
    pub(super) records: Rc<RefCell<Vec<Attempt>>>,
    pub(super) failure: Option<Failure>,
    overflow: Rc<std::cell::Cell<bool>>,
}

impl Registration {
    pub(super) fn new(controller: &ICoreWebView2Controller, policy: Policy) -> Self {
        let records = Rc::new(RefCell::new(Vec::with_capacity(64)));
        let overflow = Rc::new(std::cell::Cell::new(false));
        let output = Rc::clone(&records);
        let lost = Rc::clone(&overflow);
        // Type erasure is required by the SDK's runtime callback ABI.
        let handler = AcceleratorKeyPressedEventHandler::create(Box::new(move |_, args| {
            let mut attempt = Attempt::default();
            attempt.failure = attempt.capture(args.as_ref(), policy).err();
            if let Ok(mut records) = output.try_borrow_mut() {
                if records.len() < 64 {
                    records.push(attempt);
                } else {
                    lost.set(true);
                }
            } else {
                lost.set(true);
            }
            // The caller inspects the retained failure outside this COM callback.
            Ok(())
        }));
        let mut token = 0;
        // SAFETY: handler and controller stay owned until removal and host close.
        let result = unsafe { controller.add_AcceleratorKeyPressed(&handler, &mut token) };
        let succeeded = result.is_ok();
        Self {
            controller: controller.clone(),
            _handler: handler,
            token: succeeded.then_some(token),
            records,
            failure: checked("add_AcceleratorKeyPressed", result).err(),
            overflow,
        }
    }

    pub(super) fn validate(&self) -> io::Result<()> {
        if let Some(error) = &self.failure {
            return Err(io::Error::other(error.clone()));
        }
        if let Some(error) = self
            .records
            .borrow()
            .iter()
            .find_map(|record| record.failure.as_ref())
        {
            return Err(io::Error::other(error.clone()));
        }
        if self.overflow.get() {
            return Err(io::Error::other(
                "fixture callback storage exhausted or contended",
            ));
        }
        Ok(())
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        if let Some(token) = self.token {
            // SAFETY: the token belongs to this still-live retained controller.
            unsafe { self.controller.remove_AcceleratorKeyPressed(token) }
                .map_err(io::Error::other)?;
            self.token = None;
        }
        self.validate()
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error, "fixture accelerator removal failed");
        }
    }
}
