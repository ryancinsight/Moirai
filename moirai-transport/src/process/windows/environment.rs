//! Windows environment blocks sorted with the OS ordinal case-insensitive order.
use super::super::{ProcessError, ProcessResult, ProcessSpec};
use super::{ffi, os_error};
use std::{cmp::Ordering, ffi::OsStr, os::windows::ffi::OsStrExt};

pub(super) fn encode(spec: &ProcessSpec) -> ProcessResult<Option<Vec<u16>>> {
    if !spec.clear_environment && spec.envs.is_empty() {
        return Ok(None);
    }
    let mut entries = Vec::new();
    if !spec.clear_environment {
        for (key, value) in std::env::vars_os() {
            insert(&mut entries, &key, &value)?;
        }
    }
    for (key, value) in &spec.envs {
        if key.encode_wide().any(|unit| unit == 61) {
            return Err(ProcessError::InvalidSpecification);
        }
        insert(&mut entries, key, value)?;
    }
    let mut block = Vec::new();
    for (key, value) in entries {
        block.extend(key);
        block.push(61);
        block.extend(value);
        block.push(0);
    }
    block.push(0);
    if block.len() == 1 {
        block.push(0);
    }
    Ok(Some(block))
}
fn insert(
    entries: &mut Vec<(Vec<u16>, Vec<u16>)>,
    key: &OsStr,
    value: &OsStr,
) -> ProcessResult<()> {
    let key: Vec<_> = key.encode_wide().collect();
    let value: Vec<_> = value.encode_wide().collect();
    if key.is_empty() || key.contains(&0) || value.contains(&0) {
        return Err(ProcessError::InvalidSpecification);
    }
    let mut index = 0;
    while let Some((existing, _)) = entries.get(index) {
        match compare(&key, existing)? {
            Ordering::Less => break,
            Ordering::Equal => {
                entries[index] = (key, value);
                return Ok(());
            }
            Ordering::Greater => index += 1,
        }
    }
    entries.insert(index, (key, value));
    Ok(())
}
fn compare(left: &[u16], right: &[u16]) -> ProcessResult<Ordering> {
    let left_count = i32::try_from(left.len()).map_err(|_| ProcessError::InvalidSpecification)?;
    let right_count = i32::try_from(right.len()).map_err(|_| ProcessError::InvalidSpecification)?;
    // SAFETY: both counted UTF-16 slices remain live; ignore-case selects the
    // Windows environment's locale-independent ordinal key equivalence.
    match unsafe {
        ffi::CompareStringOrdinal(left.as_ptr(), left_count, right.as_ptr(), right_count, 1)
    } {
        1 => Ok(Ordering::Less),
        2 => Ok(Ordering::Equal),
        3 => Ok(Ordering::Greater),
        _ => Err(os_error(ProcessError::SpawnFailed)),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn environment_overrides_ignore_case_and_are_sorted() {
        let spec = ProcessSpec::new("program.exe")
            .env_clear()
            .env("z", "last")
            .env("PATH", "old")
            .env("path", "new")
            .env("a", "first");
        let block = encode(&spec)
            .expect("environment encoding")
            .expect("explicit block");
        assert_eq!(
            block,
            "a=first\0path=new\0z=last\0\0"
                .encode_utf16()
                .collect::<Vec<_>>()
        );
    }
}
