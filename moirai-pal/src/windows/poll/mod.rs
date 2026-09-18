//! Windows readiness reactor backed by `WSAPoll`.
//!
//! The IOCP completion model signals completions of *posted overlapped
//! operations*, not socket *readiness*, so it cannot drive the readiness-based
//! futures in [`crate::net`] (which do a non-blocking syscall and register a
//! waker on `WouldBlock`). This reactor uses `WSAPoll` — the Windows analogue of
//! `poll(2)` — to report which registered sockets are readable/writable, which is
//! exactly the readiness signal those futures need.
//!
//! PAL socket registrations retain a weak OS-socket owner. Each poll snapshot
//! upgrades that owner and holds the strong lease through `WSAPoll`, excluding
//! concurrent `closesocket`; an owner retired before snapshot acquisition is
//! invalidated without entering the kernel call. Raw registrations remain
//! caller-owned and a closed raw socket surfaces as `POLLNVAL`. Every
//! invalidation carries its registration generation, so a delayed event cannot
//! consume a newer registration for a reused raw socket value.

mod polling;
mod reactor_impl;
mod registration;
mod types;

#[cfg(test)]
mod tests;

pub use types::WsaPollReactor;
