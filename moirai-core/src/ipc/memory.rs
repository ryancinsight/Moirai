//! OS shared-memory segments: POSIX `shm_open`/`mmap` and Win32 file mappings.
//!
//! [`SharedMemory`] owns one mapping of `size` bytes at `ptr`, valid until
//! `Drop` unmaps it. It is the raw substrate under
//! [`SharedQueue`](super::SharedQueue); the two contracts below are what make
//! its safe API sound.
//!
//! # The mapping must cover `size`
//!
//! `as_slice`/`as_mut_slice` build a slice of exactly `size` bytes, so the
//! mapping has to be backed for all of them. The two platforms differ in who
//! enforces that, and the difference is why `open` is written the way it is:
//!
//! - **Windows** enforces it. `MapViewOfFile` requires the requested view to lie
//!   within the mapping object, and fails otherwise, so an oversized `open` is
//!   rejected by the OS.
//! - **POSIX does not.** `mmap` accepts a length running past the end of the
//!   object; the pages beyond it simply are not backed, and touching them raises
//!   `SIGBUS`. `open` therefore `fstat`s the descriptor and rejects a segment
//!   smaller than `size` itself — without that check a caller could open an
//!   existing segment under a too-large `size` and get a slice that faults on
//!   read. `create` needs no such check because its `ftruncate` sets the object
//!   to exactly `size`.
//!
//! The POSIX check reads the size from the descriptor it goes on to map, so it
//! cannot be defeated by re-resolving the name. It does not cover a process that
//! *shrinks* the object with `ftruncate` after the check — an act that already
//! invalidates every existing mapping of that segment, and which POSIX gives no
//! way to exclude.
//!
//! # Cross-process aliasing is the caller's contract
//!
//! A segment is shared by construction: another process holding the same name
//! may write it at any time. `&[u8]` and `&mut [u8]` promise Rust that no such
//! concurrent write happens, and no OS primitive here can enforce that. The safe
//! accessors therefore carry a contract the caller must uphold — either be the
//! only party touching the bytes for the borrow, or coordinate externally.
//! [`SharedQueue`](super::SharedQueue) is the coordinated wrapper: it never hands
//! out a slice, reaching the bytes through raw pointers with the atomic head/tail
//! protocol in its metadata header instead.
//!
//! Ownership is separate from mapping: `owner` records who created the segment,
//! so only the creator `shm_unlink`s the name on drop. Every handle unmaps its
//! own view and closes its own descriptor regardless.

use super::error::{last_os_error, IpcError};
use core::slice;

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// Raw Win32 file-mapping bindings
#[cfg(windows)]
mod win {
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const FILE_MAP_ALL_ACCESS: u32 = 0x000F_001F;
    pub const INVALID_HANDLE_VALUE: usize = usize::MAX;

    extern "system" {
        pub fn CreateFileMappingW(
            file: usize,
            attributes: *mut core::ffi::c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> usize;
        pub fn OpenFileMappingW(desired_access: u32, inherit: i32, name: *const u16) -> usize;
        pub fn MapViewOfFile(
            mapping: usize,
            desired_access: u32,
            offset_high: u32,
            offset_low: u32,
            size: usize,
        ) -> *mut core::ffi::c_void;
        pub fn UnmapViewOfFile(address: *const core::ffi::c_void) -> i32;
        pub fn CloseHandle(handle: usize) -> i32;
    }

    pub fn wide_name(name: &str) -> Vec<u16> {
        name.trim_start_matches('/')
            .encode_utf16()
            .chain(core::iter::once(0))
            .collect()
    }
}

/// Shared memory segment for zero-copy IPC
pub struct SharedMemory {
    /// Memory-mapped region
    pub(crate) ptr: *mut u8,
    /// Size of the shared memory
    pub(crate) size: usize,
    /// File descriptor (Unix) or handle (Windows)
    #[cfg(unix)]
    fd: RawFd,
    #[cfg(windows)]
    handle: usize,
    /// Whether this instance owns the memory
    #[cfg_attr(windows, allow(dead_code))]
    owner: bool,
    /// Name of the segment (Unix only, to allow `shm_unlink` on drop)
    #[cfg(unix)]
    name: Option<std::ffi::CString>,
}

#[cfg(unix)]
fn unix_mapping_length(size: usize) -> Result<libc::off_t, IpcError> {
    if size == 0 {
        return Err(IpcError::InvalidArgument);
    }

    libc::off_t::try_from(size).map_err(|_| IpcError::InvalidArgument)
}

// SAFETY: the mapping is process-wide, not thread-owned — `ptr` stays valid on
// any thread for the lifetime of this handle, and neither the descriptor nor the
// handle is thread-affine. `Send` therefore moves a still-valid mapping.
unsafe impl Send for SharedMemory {}

// SAFETY: `&SharedMemory` reaches only `as_slice`, which performs no interior
// mutation, so sharing the handle across threads adds no race the type does not
// already have. In-process `&mut` access is excluded by the borrow checker via
// `as_mut_slice(&mut self)`; concurrent writes from *other processes* are outside
// what any impl here can enforce and are the caller's contract (see module docs).
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Create a new shared memory segment
    #[cfg(unix)]
    pub fn create(name: &str, size: usize) -> Result<Self, IpcError> {
        use std::ffi::CString;

        let mapping_length = unix_mapping_length(size)?;
        let c_name = CString::new(name).map_err(|_| IpcError::InvalidArgument)?;

        // SAFETY: `c_name` is NUL-terminated, `mapping_length` is positive and
        // representable as `off_t`, and every failed descriptor/mapping path is
        // closed before returning. The successful mapping owns `size` bytes
        // until `Drop` unmaps it.
        unsafe {
            use std::ptr::null_mut;
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);

            if fd < 0 {
                return Err(last_os_error());
            }

            if libc::ftruncate(fd, mapping_length) < 0 {
                libc::close(fd);
                return Err(last_os_error());
            }

            let ptr = libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(last_os_error());
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                fd,
                owner: true,
                name: Some(c_name),
            })
        }
    }

    /// Open an existing shared memory segment.
    ///
    /// Fails with [`IpcError::InvalidArgument`] if the segment is smaller than
    /// `size`. `mmap` accepts a length past the end of the object, but the pages
    /// beyond it are not backed: reading them raises `SIGBUS`, so an unchecked
    /// mapping would hand out an `as_slice` that faults instead of reading.
    #[cfg(unix)]
    pub fn open(name: &str, size: usize) -> Result<Self, IpcError> {
        use std::ffi::CString;

        let mapping_length = unix_mapping_length(size)?;
        let c_name = CString::new(name).map_err(|_| IpcError::InvalidArgument)?;

        // SAFETY: `c_name` is NUL-terminated and `mapping_length` is positive and
        // representable as `off_t`. The descriptor is closed on every failure
        // path, and the mapping is only kept once `fstat` proves the object
        // covers all `size` bytes.
        unsafe {
            use std::ptr::null_mut;
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0);

            if fd < 0 {
                return Err(last_os_error());
            }

            let mut segment = core::mem::MaybeUninit::<libc::stat>::uninit();
            if libc::fstat(fd, segment.as_mut_ptr()) < 0 {
                let error = last_os_error();
                libc::close(fd);
                return Err(error);
            }

            // `fstat` succeeded, so the OS initialized the struct.
            if segment.assume_init().st_size < mapping_length {
                libc::close(fd);
                return Err(IpcError::InvalidArgument);
            }

            let ptr = libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(last_os_error());
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                fd,
                owner: false,
                name: None,
            })
        }
    }

    /// Create a new shared memory segment
    #[cfg(windows)]
    pub fn create(name: &str, size: usize) -> Result<Self, IpcError> {
        if size == 0 {
            return Err(IpcError::InvalidArgument);
        }
        let wide = win::wide_name(name);

        // justification: Win32 `CreateFileMappingW` takes the mapping size as a
        // (high DWORD, low DWORD) pair. `size_high` carries the top 32 bits and
        // `size_low` the bottom 32; the `as u32` truncation on `size_low` is the
        // API contract, not a lossy conversion.
        #[allow(clippy::cast_possible_truncation)]
        let size_low = size as u32;
        let size_high = (size as u64 >> 32) as u32;
        // SAFETY: `wide` is a NUL-terminated UTF-16 name that outlives the call,
        // and the size pair describes `size` bytes. The handle is closed if the
        // view fails to map, so no failure path leaks it.
        unsafe {
            let handle = win::CreateFileMappingW(
                win::INVALID_HANDLE_VALUE,
                core::ptr::null_mut(),
                win::PAGE_READWRITE,
                size_high,
                size_low,
                wide.as_ptr(),
            );
            if handle == 0 {
                return Err(last_os_error());
            }

            let ptr = win::MapViewOfFile(handle, win::FILE_MAP_ALL_ACCESS, 0, 0, size);
            if ptr.is_null() {
                let error = last_os_error();
                win::CloseHandle(handle);
                return Err(error);
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                handle,
                owner: true,
            })
        }
    }

    /// Open an existing shared memory segment
    #[cfg(windows)]
    pub fn open(name: &str, size: usize) -> Result<Self, IpcError> {
        if size == 0 {
            return Err(IpcError::InvalidArgument);
        }
        let wide = win::wide_name(name);

        // SAFETY: `wide` is a NUL-terminated UTF-16 name that outlives the call.
        // `MapViewOfFile` rejects a view larger than the mapping object, so a
        // successful return proves all `size` bytes are backed; the handle is
        // closed on the failure path.
        unsafe {
            let handle = win::OpenFileMappingW(win::FILE_MAP_ALL_ACCESS, 0, wide.as_ptr());
            if handle == 0 {
                return Err(last_os_error());
            }

            let ptr = win::MapViewOfFile(handle, win::FILE_MAP_ALL_ACCESS, 0, 0, size);
            if ptr.is_null() {
                let error = last_os_error();
                win::CloseHandle(handle);
                return Err(error);
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                handle,
                owner: false,
            })
        }
    }

    /// Get a slice of the shared memory.
    ///
    /// The borrow assumes no other process writes the segment while it is held
    /// (see the module docs); use [`SharedQueue`](super::SharedQueue) when
    /// another process is an active writer.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live mapping of `size` bytes — `create` sized the
        // object with `ftruncate` and `open` rejected a segment smaller than
        // `size` — so every byte is backed and readable, and `u8` needs no
        // alignment beyond the page-aligned base. Absence of a concurrent
        // cross-process writer is the caller's contract.
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get a mutable slice of the shared memory.
    ///
    /// The borrow assumes no other process reads or writes the segment while it
    /// is held (see the module docs).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and `&mut self` excludes any other in-process
        // borrow of the same mapping for the lifetime of the returned slice.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        // SAFETY: `&mut self` in `drop` is exclusive, and `ptr`/`size` are the
        // exact base and length this handle mapped, so `munmap` releases its own
        // view and nothing else. The name is unlinked only by the creator, so a
        // handle from `open` never removes a segment others still use.
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            libc::close(self.fd);
            if self.owner {
                if let Some(ref name) = self.name {
                    libc::shm_unlink(name.as_ptr());
                }
            }
        }
        // SAFETY: `&mut self` in `drop` is exclusive; `ptr` is the base this
        // handle received from `MapViewOfFile` and `handle` the mapping it came
        // from, so each is released exactly once. The mapping object outlives
        // this close while any other process still holds it open.
        #[cfg(windows)]
        unsafe {
            win::UnmapViewOfFile(self.ptr as *const core::ffi::c_void);
            win::CloseHandle(self.handle);
        }
    }
}
