//! POSIX process-group containment.
//!
//! A contained child is spawned as the leader of a new process group, so
//! every descendant it creates normally shares that group and one signal
//! reaches the whole tree. Completion is observed with `waitid(WNOWAIT)`,
//! which reports the leader's exit without reaping it: the unreaped leader
//! keeps its PID — and therefore the group ID — reserved, so a later group
//! signal cannot reach an unrelated process that reused the number. The
//! leader is reaped only after the group has been signalled.
//!
//! Descendants that deliberately leave the group (`setsid`, `setpgid`)
//! escape, as processes granted breakaway escape a Windows job; and unlike a
//! kill-on-close job, the group is not signalled if this process itself dies
//! without dropping its handle.

use std::{io, process::ExitStatus};

/// Observes the exit of leader `pid` without reaping it.
///
/// # Errors
/// Returns the OS error when the wait fails.
pub(super) fn observe_exit(pid: u32) -> io::Result<Option<ExitStatus>> {
    let id = libc::id_t::from(pid);
    // SAFETY: siginfo_t is plain data for which all-zero bytes are valid.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a live, writable siginfo_t; P_PID with the child's id
    // only observes this process's own child; WNOWAIT leaves it unreaped.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            id,
            &raw mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    // With WNOHANG, a still-running child leaves si_pid zero.
    // SAFETY: waitid succeeded, so the child fields of `info` are initialized.
    let (reported, status) = unsafe { (info.si_pid(), info.si_status()) };
    if reported == 0 {
        return Ok(None);
    }
    Ok(Some(raw_status(info.si_code, status)))
}

/// Sends `SIGKILL` to every member of group `pgid`. A group with no
/// remaining members is already terminated.
///
/// # Errors
/// Returns the OS error for any failure other than an empty group.
pub(super) fn kill_group(pgid: u32) -> io::Result<()> {
    let group =
        libc::pid_t::try_from(pgid).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: killpg only sends a signal; `group` names a group this process
    // created and still reserves through its unreaped leader.
    if unsafe { libc::killpg(group, libc::SIGKILL) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

/// Encodes a `waitid` result as the raw `wait` status `ExitStatus` wraps.
fn raw_status(code: libc::c_int, status: libc::c_int) -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    let raw = match code {
        libc::CLD_EXITED => (status & 0xff) << 8,
        libc::CLD_DUMPED => (status & 0x7f) | 0x80,
        _ => status & 0x7f,
    };
    ExitStatus::from_raw(raw)
}

#[cfg(test)]
mod tests {
    use super::raw_status;

    #[test]
    fn waitid_results_round_trip_through_exit_status() {
        use std::os::unix::process::ExitStatusExt;
        let exited = raw_status(libc::CLD_EXITED, 7);
        assert_eq!(exited.code(), Some(7));
        let killed = raw_status(libc::CLD_KILLED, libc::SIGKILL);
        assert_eq!(killed.code(), None);
        assert_eq!(killed.signal(), Some(libc::SIGKILL));
        let dumped = raw_status(libc::CLD_DUMPED, libc::SIGSEGV);
        assert_eq!(dumped.signal(), Some(libc::SIGSEGV));
        assert!(dumped.core_dumped());
    }
}
