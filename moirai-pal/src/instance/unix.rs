//! Unix instance claims: an advisory lock plus a socket in a private directory.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, ErrorKind};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::{INSTANCE_IO_TIMEOUT, InstanceName, read_frame, write_frame};

/// Longest socket path every supported Unix accepts (macOS `sun_path`).
const MAX_SOCKET_PATH_BYTES: usize = 103;
/// Pause between connection attempts while a new primary starts listening.
const CONNECT_RETRY: Duration = Duration::from_millis(10);

pub(super) enum Role {
    Primary(Primary),
    Secondary(Secondary),
}

#[derive(Debug)]
pub(super) struct Primary {
    listener: UnixListener,
    socket: PathBuf,
    // Held for the process lifetime; the kernel releases it on exit.
    _lock: File,
}

#[derive(Debug)]
pub(super) struct Secondary {
    stream: UnixStream,
}

pub(super) fn claim(name: &InstanceName) -> io::Result<Role> {
    let directory = private_directory()?;
    let socket = directory.join(format!("{}.sock", name.as_str()));
    if socket.as_os_str().len() > MAX_SOCKET_PATH_BYTES {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "instance socket path exceeds the platform limit",
        ));
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(directory.join(format!("{}.lock", name.as_str())))?;
    // SAFETY: the descriptor belongs to `lock`, which outlives the call.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        match fs::remove_file(&socket) {
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        let listener = UnixListener::bind(&socket)?;
        listener.set_nonblocking(true)?;
        return Ok(Role::Primary(Primary {
            listener,
            socket,
            _lock: lock,
        }));
    }
    let error = io::Error::last_os_error();
    if error.kind() != ErrorKind::WouldBlock {
        return Err(error);
    }
    connect(&socket).map(|stream| Role::Secondary(Secondary { stream }))
}

/// Connects to the primary, waiting while it moves from locking to listening.
fn connect(socket: &Path) -> io::Result<UnixStream> {
    let deadline = Instant::now() + INSTANCE_IO_TIMEOUT;
    loop {
        match UnixStream::connect(socket) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::NotFound | ErrorKind::ConnectionRefused
                ) && Instant::now() < deadline =>
            {
                std::thread::sleep(CONNECT_RETRY);
            }
            Err(error) if matches!(error.kind(), ErrorKind::NotFound) => {
                return Err(io::Error::new(
                    ErrorKind::TimedOut,
                    "instance primary did not start listening",
                ));
            }
            Err(error) => return Err(error),
        }
    }
}

/// The per-user directory holding claims, created owner-only.
///
/// `XDG_RUNTIME_DIR` is used when set; otherwise `/tmp`, whose short path
/// keeps socket names within `sun_path`. An existing directory must be a
/// real directory owned by this user and closed to everyone else.
fn private_directory() -> io::Result<PathBuf> {
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let directory = base.join(format!("moirai-instance-{uid}"));
    match DirBuilder::new().mode(0o700).create(&directory) {
        Err(error) if error.kind() != ErrorKind::AlreadyExists => return Err(error),
        _ => {}
    }
    let metadata = fs::symlink_metadata(&directory)?;
    if !metadata.file_type().is_dir() || metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "instance directory is not private to this user",
        ));
    }
    Ok(directory)
}

impl Primary {
    pub(super) fn try_receive(&mut self) -> io::Result<Option<Vec<u8>>> {
        let mut stream = match self.listener.accept() {
            Ok((stream, _)) => stream,
            Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
            Err(error) => return Err(error),
        };
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(INSTANCE_IO_TIMEOUT))?;
        read_frame(&mut stream).map(Some).map_err(|error| {
            if error.kind() == ErrorKind::WouldBlock {
                io::Error::new(ErrorKind::TimedOut, "instance sender stalled")
            } else {
                error
            }
        })
    }
}

impl Drop for Primary {
    fn drop(&mut self) {
        // The lock is still held, so no other primary owns this path yet.
        let _ = fs::remove_file(&self.socket);
    }
}

impl Secondary {
    pub(super) fn send(mut self, message: &[u8]) -> io::Result<()> {
        self.stream.set_write_timeout(Some(INSTANCE_IO_TIMEOUT))?;
        write_frame(&mut self.stream, message)?;
        self.stream.shutdown(std::net::Shutdown::Write)
    }
}
