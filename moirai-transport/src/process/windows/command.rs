//! Bounded Windows command-line encoding under the MSVC argument rules.
//!
//! Reference: <https://learn.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments>.
use super::super::{ProcessError, ProcessResult};
use std::{os::windows::ffi::OsStrExt, path::Path};

// CreateProcessW's maximum includes the terminating UTF-16 zero.
const COMMAND_CAPACITY: usize = 32_767;

pub(super) fn encode(
    binary: &Path,
    arguments: &[std::ffi::OsString],
) -> ProcessResult<(Vec<u16>, Vec<u16>)> {
    let mut application = Vec::new();
    for unit in binary.as_os_str().encode_wide() {
        append(&mut application, unit)?;
    }
    if application.is_empty() || application.contains(&34) {
        return Err(invalid());
    }
    let mut command = Vec::new();
    append(&mut command, 34)?;
    for &unit in &application {
        append(&mut command, unit)?;
    }
    append(&mut command, 34)?;
    for argument in arguments {
        append(&mut command, 32)?;
        let quoted = argument.is_empty()
            || argument
                .encode_wide()
                .any(|unit| matches!(unit, 9 | 32 | 34));
        if quoted {
            append(&mut command, 34)?;
        }
        let mut slashes = 0;
        for unit in argument.encode_wide() {
            if unit == 92 {
                slashes += 1;
            } else {
                for _ in 0..slashes {
                    append(&mut command, 92)?;
                }
                if unit == 34 {
                    for _ in 0..=slashes {
                        append(&mut command, 92)?;
                    }
                }
                append(&mut command, unit)?;
                slashes = 0;
            }
            // Bound even a backslash-only argument before its closing quote.
            if slashes >= COMMAND_CAPACITY {
                return Err(invalid());
            }
        }
        for _ in 0..slashes {
            append(&mut command, 92)?;
            if quoted {
                append(&mut command, 92)?;
            }
        }
        if quoted {
            append(&mut command, 34)?;
        }
    }
    application.push(0);
    command.push(0);
    Ok((application, command))
}
fn append(output: &mut Vec<u16>, unit: u16) -> ProcessResult<()> {
    if unit == 0 || output.len() >= COMMAND_CAPACITY - 1 {
        return Err(invalid());
    }
    output.push(unit);
    Ok(())
}
fn invalid() -> ProcessError {
    ProcessError::InvalidSpecification
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quoted_arguments_follow_msvc_backslash_rules() {
        for (argument, expected) in [
            ("", "\"program.exe\" \"\""),
            ("a b", "\"program.exe\" \"a b\""),
            ("a\\", "\"program.exe\" a\\"),
            ("a\"b", "\"program.exe\" \"a\\\"b\""),
            ("a\\\"b", "\"program.exe\" \"a\\\\\\\"b\""),
        ] {
            let (application, command) =
                encode(Path::new("program.exe"), &[argument.into()]).expect("command encoding");
            assert_eq!(
                application,
                "program.exe\0".encode_utf16().collect::<Vec<_>>()
            );
            assert_eq!(
                command,
                format!("{expected}\0").encode_utf16().collect::<Vec<_>>()
            );
        }
    }
}
