//! The [`Reactor`] trait surface for [`WsaPollReactor`].

use std::io;
use std::time::Duration;

use crate::reactor::registration::PlatformUpdateFailure;
use crate::{Event, Interest, RawFd, Reactor};

use super::types::{WsaPollReactor, lock_mutex};

impl Reactor for WsaPollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        self.replace_registration(fd, interest)
            .map_err(PlatformUpdateFailure::into_error)
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        lock_mutex(&self.registrations).remove(fd as usize);
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        self.poll_events_with(timeout, |event, _generation, _invalidated| event)
    }

    fn wake(&self) -> io::Result<()> {
        self.wake.send_to(&[1u8], self.wake_addr).map(|_| ())
    }
}
