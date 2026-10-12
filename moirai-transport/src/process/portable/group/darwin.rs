//! Darwin rejects signalling a process group containing only exited members.
//! `proc_listpgrppids` enumerates the group, including unreaped zombies;
//! `PROC_PIDT_SHORTBSDINFO` queries live members without same-user privileges.
//! XNU returns ESRCH before permission checks for an exited process. A positive
//! live record prevents completion; denied or incomplete queries remain errors.
//! The unreaped group leader reserves the group identity throughout this query.
//!
//! Kernel contracts: xnu/bsd/kern/kern_sig.c::killpg1 and
//! xnu/bsd/kern/proc_info.c::proc_pidinfo (Apple open-source distribution).

use std::{
    io,
    mem::{MaybeUninit, size_of},
};

pub(super) fn completed(group: libc::pid_t) -> io::Result<bool> {
    // SAFETY: libproc receives a group identity and no output buffer; this
    // call returns an upper estimate of the number of PID slots required.
    let estimate = unsafe { libc::proc_listpgrppids(group, std::ptr::null_mut(), 0) };
    if estimate <= 0 {
        return Err(io::Error::last_os_error());
    }
    let capacity = usize::try_from(estimate).map_err(io::Error::other)?;
    let size = capacity
        .checked_mul(size_of::<libc::pid_t>())
        .ok_or_else(|| io::Error::other("process-group allocation overflow"))?;
    let bytes = libc::c_int::try_from(size).map_err(io::Error::other)?;
    let mut pids = Vec::<libc::pid_t>::new();
    pids.try_reserve_exact(capacity).map_err(io::Error::other)?;
    pids.resize(capacity, 0);
    // SAFETY: pids owns bytes of writable, aligned PID storage. libproc
    // returns the initialized PID count, rather than the byte count.
    let written = unsafe { libc::proc_listpgrppids(group, pids.as_mut_ptr().cast(), bytes) };
    if written <= 0 {
        return Err(io::Error::last_os_error());
    }
    let count = usize::try_from(written).map_err(io::Error::other)?;
    if count >= capacity {
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "process-group enumeration filled its buffer",
        ));
    }
    for &pid in &pids[..count] {
        let mut info = MaybeUninit::<libc::proc_bsdshortinfo>::uninit();
        let bytes = libc::c_int::try_from(size_of::<libc::proc_bsdshortinfo>())
            .map_err(io::Error::other)?;
        // SAFETY: info is writable, correctly aligned storage for the exact
        // short BSD record requested. No record is read before a full write.
        let written = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDT_SHORTBSDINFO,
                0,
                info.as_mut_ptr().cast(),
                bytes,
            )
        };
        if written == 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                continue;
            }
            return Err(error);
        }
        if written != bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Darwin returned an incomplete process record",
            ));
        }
        // SAFETY: proc_pidinfo wrote a complete proc_bsdshortinfo above.
        let info = unsafe { info.assume_init() };
        if info.pbsi_pgid == u32::try_from(group).map_err(io::Error::other)? {
            return Ok(false);
        }
    }
    Ok(true)
}
