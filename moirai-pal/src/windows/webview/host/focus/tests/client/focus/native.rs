//! One real SetFocus through the host's synchronous controller focus owner.

use super::Observations;
use std::{error::Error, fmt, io, time::Instant};
use windows::Win32::{
    Foundation::{ERROR_SUCCESS, GetLastError, HWND, SetLastError},
    System::Threading::{AttachThreadInput, GetCurrentThreadId},
    UI::{
        Input::KeyboardAndMouse::GetFocus,
        WindowsAndMessaging::{
            GetForegroundWindow, GetWindowThreadProcessId, IsChild, MSG, PM_NOREMOVE, PeekMessageW,
        },
    },
};

// The generated Result<HWND> wrapper rejects a null *previous* focus handle,
// which cannot represent successful entry from a previously unfocused queue.
// Preserve the native return and immediate error separately, then verify the
// real delegated focus. HWND is the SDK's transparent native handle ABI.
#[link(name = "user32")]
unsafe extern "system" {
    #[link_name = "SetFocus"]
    fn native_set_focus(parent: HWND) -> HWND;
}

#[derive(Debug)]
pub(super) struct Entry {
    pub(super) parent: HWND,
    pub(super) previous: HWND,
    pub(super) recipient: HWND,
    pub(super) client_thread: u32,
    pub(super) owner_thread: u32,
    pub(super) owner_process: u32,
}

struct Attachment {
    client: u32,
    owner: u32,
    active: bool,
}

impl Attachment {
    fn new(client: u32, owner: u32, observations: &mut Observations) -> io::Result<Self> {
        let mut attachment = Self {
            client,
            owner,
            active: false,
        };
        if client != owner {
            // SAFETY: both actual thread IDs have Win32 message queues. The
            // owned foreground parent is on this interactive desktop.
            observations
                .operation("AttachThreadInput_parent", || {
                    let result = unsafe { AttachThreadInput(client, owner, true).ok() };
                    attachment.active = result.is_ok();
                    result
                })
                .map_err(io::Error::other)?;
        }
        Ok(attachment)
    }

    fn detach(mut self, observations: &mut Observations) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        observations
            .operation("DetachThreadInput_parent", || {
                // Disarm at actual cleanup entry; a panic in the observation's
                // start record still leaves Drop responsible for the detach.
                self.active = false;
                unsafe { AttachThreadInput(self.client, self.owner, false).ok() }
            })
            .map_err(io::Error::other)
    }
}

impl Drop for Attachment {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            // SAFETY: this guard owns the successful attachment. This path
            // only handles unwinding before explicit error-preserving cleanup.
            if let Err(error) = unsafe { AttachThreadInput(self.client, self.owner, false).ok() } {
                tracing::error!(client_thread=self.client,owner_thread=self.owner,%error,
                    "native parent input-queue detach failed during unwinding");
            }
        }
    }
}

#[derive(Debug)]
struct EntryAndDetachFailure {
    entry: io::Error,
    detach: io::Error,
}

impl fmt::Display for EntryAndDetachFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native parent entry: {}; input-queue detach: {}",
            self.entry, self.detach
        )
    }
}

impl Error for EntryAndDetachFailure {
    // The standard cold diagnostic error ABI requires error-type erasure;
    // both concrete original errors remain owned above.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.entry)
    }
}

pub(super) fn enter(
    handle: usize,
    deadline: Instant,
    observations: &mut Observations,
) -> io::Result<Entry> {
    let parent = HWND(handle as *mut _);
    let client_thread = unsafe { GetCurrentThreadId() };
    let mut message = MSG::default();
    // SAFETY: the writable MSG belongs to this MTA. A non-removing peek creates
    // its real message queue and consumes no posted keyboard input.
    let message_pending = observations
        .operation("PeekMessageW_create_queue", || unsafe {
            PeekMessageW(&mut message, None, 0, 0, PM_NOREMOVE)
        })
        .as_bool();
    observations.record(
        serde_json::json!({"kind":"native_client_queue", "message_pending":message_pending}),
    );
    let mut owner_process = 0;
    let (owner_thread, error) = observations.operation("GetWindowThreadProcessId_parent", || {
        let thread = unsafe { GetWindowThreadProcessId(parent, Some(&mut owner_process)) };
        (thread, (thread == 0).then(io::Error::last_os_error))
    });
    if owner_thread == 0 {
        return Err(error.expect("immediately captured native parent thread error"));
    }
    observations.record(serde_json::json!({"kind":"native_parent_request",
        "parent":format!("{parent:?}"),"owner_thread_id":owner_thread,
        "owner_process_id":owner_process,"client_thread_id":client_thread}));
    let attachment = Attachment::new(client_thread, owner_thread, observations)?;
    let entry = attached_entry(
        parent,
        client_thread,
        owner_thread,
        owner_process,
        deadline,
        observations,
    );
    let detach = attachment.detach(observations);
    match (entry, detach) {
        (Ok(entry), Ok(())) => Ok(entry),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(entry), Err(detach)) => Err(io::Error::other(EntryAndDetachFailure { entry, detach })),
    }
}

fn attached_entry(
    parent: HWND,
    client_thread: u32,
    owner_thread: u32,
    owner_process: u32,
    deadline: Instant,
    observations: &mut Observations,
) -> io::Result<Entry> {
    // SAFETY: the caller owns the parent or temporarily owns its input queue
    // attachment. Reads below observe this actual shared native focus queue.
    let foreground = observations.operation("GetForegroundWindow_parent", || unsafe {
        GetForegroundWindow()
    });
    if foreground != parent {
        return Err(io::Error::other(
            "native parent entry requires the owned foreground window",
        ));
    }
    let previous = observations.operation("SetFocus_native_parent", || {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "native parent focus admission exceeds original deadline",
            ));
        }
        // SAFETY: parent is live and belongs to this owner/attached queue. A
        // cleared error distinguishes a null previous handle from API failure;
        // the post-call descendant checks still establish actual delegation.
        unsafe {
            SetLastError(ERROR_SUCCESS);
            let previous = native_set_focus(parent);
            let error = GetLastError();
            if previous.is_invalid() {
                error.ok().map_err(io::Error::other)?;
            }
            Ok(previous)
        }
    })?;
    let recipient = observations.operation("GetFocus_native_parent", || unsafe { GetFocus() });
    let descendant = observations
        .operation("IsChild_native_parent", || unsafe {
            IsChild(parent, recipient)
        })
        .as_bool();
    if recipient.is_invalid() || recipient == parent || !descendant {
        return Err(io::Error::other(format!(
            "native parent did not delegate focus: parent={parent:?}, recipient={recipient:?}"
        )));
    }
    Ok(Entry {
        parent,
        previous,
        recipient,
        client_thread,
        owner_thread,
        owner_process,
    })
}
