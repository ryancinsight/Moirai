//! Launch-at-login entries in the current user's `Run` key.
//!
//! This is the Windows half of Tauri's autostart plugin. The API is limited
//! to named values under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`:
//! it reads, writes and removes one application's command line there and
//! can reach no other registry location.

use std::io;

use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, WIN32_ERROR};
use windows::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SZ,
    RRF_RT_REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteValueW, RegGetValueW, RegSetValueExW,
};
use windows::core::PCWSTR;

/// Maximum UTF-16 units in an entry name.
pub const MAX_STARTUP_NAME_UNITS: usize = 64;
/// Maximum UTF-16 units in an entry's command line.
pub const MAX_STARTUP_COMMAND_UNITS: usize = 2_048;

const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";

/// Writes or replaces the launch-at-login command for `name`.
///
/// # Errors
/// Returns `InvalidInput` for an invalid name or command, or the registry
/// error.
pub fn set_startup_command(name: &str, command: &str) -> io::Result<()> {
    let name = encode(name, MAX_STARTUP_NAME_UNITS, true)?;
    let command = encode(command, MAX_STARTUP_COMMAND_UNITS, false)?;
    let key = RunKey::open(KEY_SET_VALUE)?;
    let bytes: Vec<u8> = command.iter().flat_map(|unit| unit.to_le_bytes()).collect();
    // SAFETY: the key is open for writing and both buffers are NUL-terminated
    // and outlive the synchronous call.
    check(unsafe { RegSetValueExW(key.0, PCWSTR(name.as_ptr()), None, REG_SZ, Some(&bytes)) })
}

/// Removes the entry for `name`; returns whether one existed.
///
/// # Errors
/// Returns `InvalidInput` for an invalid name, or the registry error.
pub fn remove_startup_command(name: &str) -> io::Result<bool> {
    let name = encode(name, MAX_STARTUP_NAME_UNITS, true)?;
    let key = RunKey::open(KEY_SET_VALUE)?;
    // SAFETY: the key is open for writing and the name is NUL-terminated.
    let status = unsafe { RegDeleteValueW(key.0, PCWSTR(name.as_ptr())) };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(false);
    }
    check(status).map(|()| true)
}

/// Reads the command registered for `name`, if any.
///
/// # Errors
/// Returns `InvalidInput` for an invalid name, `InvalidData` for a value
/// that is not UTF-16, or the registry error, which reports a value over
/// [`MAX_STARTUP_COMMAND_UNITS`] as more data than the buffer holds.
pub fn startup_command(name: &str) -> io::Result<Option<String>> {
    let name = encode(name, MAX_STARTUP_NAME_UNITS, true)?;
    let key = RunKey::open(KEY_QUERY_VALUE)?;
    let mut buffer = vec![0_u16; MAX_STARTUP_COMMAND_UNITS + 1];
    let mut bytes = u32::try_from(buffer.len() * 2).unwrap_or(u32::MAX);
    // SAFETY: the key is open for reading; `buffer` holds `bytes` writable
    // bytes and the name is NUL-terminated.
    let status = unsafe {
        RegGetValueW(
            key.0,
            PCWSTR::null(),
            PCWSTR(name.as_ptr()),
            RRF_RT_REG_SZ,
            None,
            Some(buffer.as_mut_ptr().cast()),
            Some(&mut bytes),
        )
    };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(None);
    }
    check(status)?;
    let units = usize::try_from(bytes / 2).unwrap_or(0);
    let value = &buffer[..units.min(buffer.len())];
    let value = value.strip_suffix(&[0]).unwrap_or(value);
    String::from_utf16(value)
        .map(Some)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "startup command is not UTF-16"))
}

/// The opened `Run` key, closed on drop.
struct RunKey(HKEY);

impl RunKey {
    fn open(access: windows::Win32::System::Registry::REG_SAM_FLAGS) -> io::Result<Self> {
        let path = encode(RUN_KEY, RUN_KEY.len(), false)?;
        let mut key = HKEY::default();
        // SAFETY: `path` is NUL-terminated and `key` is writable storage for
        // the synchronous call; the key is closed by `RunKey::drop`.
        check(unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR(path.as_ptr()),
                None,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                access,
                None,
                &mut key,
                None,
            )
        })?;
        Ok(Self(key))
    }
}

impl Drop for RunKey {
    fn drop(&mut self) {
        // SAFETY: the key was opened by `RunKey::open` and is closed once.
        let _ = unsafe { RegCloseKey(self.0) };
    }
}

/// Encodes NUL-free text of 1 to `limit` units, NUL-terminated. A name must
/// also avoid the `\` that separates registry paths.
fn encode(text: &str, limit: usize, name: bool) -> io::Result<Vec<u16>> {
    let units = text.encode_utf16().count();
    if units == 0
        || units > limit
        || text.chars().any(char::is_control)
        || (name && text.contains('\\'))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "startup entry text is empty, too long or contains control characters",
        ));
    }
    Ok(text.encode_utf16().chain(std::iter::once(0)).collect())
}

fn check(status: WIN32_ERROR) -> io::Result<()> {
    if status.is_ok() {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status.0 as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_STARTUP_NAME_UNITS, remove_startup_command, set_startup_command, startup_command,
    };

    #[test]
    fn entries_are_validated() {
        assert!(set_startup_command("", "x").is_err());
        assert!(set_startup_command("a\\b", "x").is_err());
        assert!(set_startup_command(&"n".repeat(MAX_STARTUP_NAME_UNITS + 1), "x").is_err());
        assert!(set_startup_command("moirai-test", "line\nbreak").is_err());
        assert!(startup_command("a\\b").is_err());
    }

    #[test]
    fn entries_round_trip_and_are_removed() {
        let name = format!("moirai-startup-test-{}", std::process::id());
        let command = r#""C:\Program Files\Moirai Test\app.exe" "--minimized""#;
        assert_eq!(startup_command(&name).expect("absent"), None);
        set_startup_command(&name, command).expect("set");
        assert_eq!(
            startup_command(&name).expect("read").as_deref(),
            Some(command)
        );
        assert!(remove_startup_command(&name).expect("remove"));
        assert!(!remove_startup_command(&name).expect("second remove"));
        assert_eq!(startup_command(&name).expect("removed"), None);
    }
}
