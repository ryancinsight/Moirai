//! Reserving the backing store of a Unix shared-memory segment.

#![cfg(unix)]

use std::os::unix::io::RawFd;

use super::error::IpcError;

/// What the kernel answered when asked to reserve a segment's backing store.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Reservation {
    /// The store is committed: every page of the segment exists.
    Reserved,
    /// This object cannot be preallocated at all, so it keeps the sparse
    /// backing `ftruncate` gave it.
    Unsupported,
    /// A signal arrived first; the request has to be reissued.
    Interrupted,
    /// The store cannot be had, under the reported `errno`.
    Failed(i32),
}

/// Classify what `posix_fallocate` returned.
///
/// Unlike the `shm_open`/`ftruncate`/`mmap` calls around it, `posix_fallocate`
/// reports through its return value and leaves `errno` untouched, so the code
/// arrives here directly instead of through `last_os_error`.
///
/// The distinction that matters is between a kernel that *will not preallocate
/// here* and one that *cannot find the store*. `create` has already rejected a
/// non-positive length, so `EINVAL` can only mean this object refuses the
/// operation, as `EOPNOTSUPP` and `ENOSYS` do on a filesystem without fallocate
/// — tmpfs before Linux 3.5, say. None of the three says anything about free
/// space, and failing creation on them would break segments that work today, so
/// they leave the mapping as sparse as it was before this call existed. Every
/// other code — `ENOSPC` and `EFBIG` above all — is the shortage the call exists
/// to surface.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(super) const fn classify_reservation(code: i32) -> Reservation {
    match code {
        0 => Reservation::Reserved,
        libc::EINTR => Reservation::Interrupted,
        libc::EOPNOTSUPP | libc::ENOSYS | libc::EINVAL => Reservation::Unsupported,
        other => Reservation::Failed(other),
    }
}

/// Reserve a whole segment's backing store, so its pages exist before anyone
/// maps them.
///
/// `ftruncate` sets the object's length and nothing more. On tmpfs the pages
/// behind that length stay sparse, so the first write to one can fail for want
/// of memory — as `SIGBUS`, inside a safe accessor, in whichever process
/// happened to touch it. Asking for the pages here turns that fault into an
/// ordinary error at creation.
///
/// Returns `Ok` when the store is committed, and equally when the kernel
/// declines to preallocate at all: the segment is still exactly `size` bytes, so
/// such a caller keeps the behavior it had before, and only a reported shortage
/// is an error. See `classify_reservation` for the split.
///
/// Unix targets outside the Linux family have no counterpart here.
/// `posix_fallocate` is absent on macOS, and this crate cannot claim tmpfs
/// semantics for platforms whose shared memory is not tmpfs, so their segments
/// keep the sparse mapping and the `SIGBUS` window that comes with it.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(super) fn reserve_backing_store(fd: RawFd, length: libc::off_t) -> Result<(), IpcError> {
    loop {
        // SAFETY: `posix_fallocate` takes `fd` as a descriptor number and two
        // integers, and answers through its return value — no pointer crosses
        // the call, so no argument reachable here can make it unsound. A
        // descriptor that is not open comes back as `EBADF`.
        let code = unsafe { libc::posix_fallocate(fd, 0, length) };

        match classify_reservation(code) {
            Reservation::Reserved | Reservation::Unsupported => return Ok(()),
            Reservation::Interrupted => {}
            Reservation::Failed(code) => return Err(IpcError::SystemError(code)),
        }
    }
}

/// Preallocation is unavailable on this target, so a segment keeps the sparse
/// backing `ftruncate` gave it. See the Linux definition for what that costs.
#[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "matches the fallible Linux signature so `create` stays platform-uniform"
)]
pub(super) fn reserve_backing_store(_fd: RawFd, _length: libc::off_t) -> Result<(), IpcError> {
    Ok(())
}
