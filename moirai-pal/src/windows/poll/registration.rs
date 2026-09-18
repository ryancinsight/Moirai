//! Socket and waiter registration, replacement, and generation bookkeeping.

use std::io;
use std::net::UdpSocket;
use std::os::windows::io::AsRawSocket;
use std::sync::Mutex;

use crate::reactor::registration::{
    PlatformUpdateFailure, PolledEvent, RegistrationGeneration, RegistrationTable,
    WaiterRegistration,
};
use crate::reactor::socket_owner::WeakSocketOwner;
use crate::{Interest, RawFd, Reactor};

use super::types::{PollBuffer, WsaPollReactor, lock_mutex};

impl WsaPollReactor {
    /// Create a reactor with an empty interest set and a fresh wake socket.
    pub fn new() -> io::Result<Self> {
        let wake = UdpSocket::bind("127.0.0.1:0")?;
        wake.set_nonblocking(true)?;
        let wake_addr = wake.local_addr()?;
        Ok(Self {
            registrations: Mutex::new(RegistrationTable::default()),
            wake,
            wake_addr,
            poll_buffer: Mutex::new(PollBuffer::default()),
            lease_buffer: Mutex::new(Vec::new()),
        })
    }

    pub(super) fn wake_socket(&self) -> usize {
        self.wake.as_raw_socket() as usize
    }

    /// Drain any pending wake datagrams (the socket is non-blocking).
    pub(super) fn drain_wake(&self) {
        let mut buf = [0u8; 64];
        while self.wake.recv(&mut buf).is_ok() {}
    }

    /// Return whether a polled event still names the registration represented
    /// by its snapshot generation.
    pub(crate) fn is_current_polled_event(&self, event: &PolledEvent) -> bool {
        lock_mutex(&self.registrations).is_current(event.event().fd as usize, event.generation())
    }

    #[cfg(test)]
    pub(crate) fn has_registration(&self, fd: RawFd) -> bool {
        lock_mutex(&self.registrations).get(fd as usize).is_some()
    }

    pub(crate) fn update_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        let socket = fd as usize;
        let mut registrations = lock_mutex(&self.registrations);
        let Some(current) = registrations.get(socket) else {
            return Err(PlatformUpdateFailure::new(
                io::Error::new(io::ErrorKind::NotFound, "WSAPoll registration is absent"),
                None,
            ));
        };
        if interest.readable || interest.writable {
            let updated = registrations.update_interest(socket, current.generation, interest);
            debug_assert!(updated, "registration remained locked during update");
        } else {
            registrations.remove(socket);
        }
        Ok(())
    }

    pub(crate) fn replace_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        self.replace_waiter_registration(fd, interest, interest)
            .map(drop)
    }

    pub(crate) fn register_waiter(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<RegistrationGeneration, PlatformUpdateFailure> {
        self.replace_waiter_registration(fd, interest, interest)
            .map(|registration| registration.generation)
    }

    pub(crate) fn register_owned_waiter(
        &self,
        fd: RawFd,
        interest: Interest,
        owner: WeakSocketOwner,
    ) -> Result<RegistrationGeneration, PlatformUpdateFailure> {
        self.replace_waiter_registration_with_owner(fd, interest, interest, Some(owner))
            .map(|registration| registration.generation)
    }

    pub(crate) fn replace_waiter_registration(
        &self,
        fd: RawFd,
        retained_interest: Interest,
        fresh_interest: Interest,
    ) -> Result<WaiterRegistration, PlatformUpdateFailure> {
        self.replace_waiter_registration_with_owner(fd, retained_interest, fresh_interest, None)
    }

    pub(crate) fn replace_owned_waiter_registration(
        &self,
        fd: RawFd,
        retained_interest: Interest,
        fresh_interest: Interest,
        owner: WeakSocketOwner,
    ) -> Result<WaiterRegistration, PlatformUpdateFailure> {
        self.replace_waiter_registration_with_owner(
            fd,
            retained_interest,
            fresh_interest,
            Some(owner),
        )
    }

    fn replace_waiter_registration_with_owner(
        &self,
        fd: RawFd,
        retained_interest: Interest,
        fresh_interest: Interest,
        owner: Option<WeakSocketOwner>,
    ) -> Result<WaiterRegistration, PlatformUpdateFailure> {
        let socket = fd as usize;
        let mut registrations = lock_mutex(&self.registrations);
        let previous = registrations.get(socket);
        let interest = if previous.is_some() {
            retained_interest
        } else {
            fresh_interest
        };
        let generation = registrations.issue_generation().map_err(|error| {
            PlatformUpdateFailure::new(error, previous.as_ref().map(|entry| entry.interest))
        })?;
        if let Some(owner) = owner {
            registrations.commit_owned(socket, interest, generation, owner);
        } else {
            registrations.commit(socket, interest, generation);
        }
        if let Err(error) = self.wake() {
            let armed_interest = if let Some(previous) = previous {
                if let Some(owner) = previous.owner {
                    registrations.commit_owned(
                        socket,
                        previous.interest,
                        previous.generation,
                        owner,
                    );
                } else {
                    registrations.commit(socket, previous.interest, previous.generation);
                }
                Some(previous.interest)
            } else {
                let removed = registrations.remove_if_current(socket, generation);
                debug_assert!(removed, "registration remained locked during wake rollback");
                None
            };
            return Err(PlatformUpdateFailure::new(error, armed_interest));
        }
        Ok(WaiterRegistration {
            generation,
            replaced_existing: previous.is_some(),
        })
    }
}
