//! Linux epoll-based async I/O reactor implementation.
//!
//! This module provides high-performance async I/O using Linux's epoll system call,
//! which is the most efficient I/O multiplexing mechanism on Linux systems.

use std::collections::HashMap;
use std::io;
use std::os::unix::io::RawFd;
use std::time::Duration;

use crate::{Reactor, Event, Interest};

/// Linux epoll-based I/O reactor.
pub struct EpollReactor {
    /// epoll file descriptor
    epoll_fd: RawFd,
    /// Event buffer for epoll_wait
    #[allow(dead_code)] // Used for future event processing per ADR requirements
    event_buffer: Vec<libc::epoll_event>,
    /// Mapping of file descriptors to their registered interests
    #[allow(dead_code)] // Used for future interest tracking per ADR requirements
    fd_interests: HashMap<RawFd, Interest>,
}

// Constants for epoll operations
const MAX_EVENTS: usize = 1024;
const EPOLL_CREATE_FLAGS: libc::c_int = libc::EPOLL_CLOEXEC;

impl EpollReactor {
    /// Create a new epoll-based reactor.
    pub fn new() -> io::Result<Self> {
        let epoll_fd = unsafe { libc::epoll_create1(EPOLL_CREATE_FLAGS) };
        
        if epoll_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        
        Ok(Self {
            epoll_fd,
            event_buffer: Vec::with_capacity(MAX_EVENTS),
            fd_interests: HashMap::new(),
        })
    }
    
    /// Convert Interest to epoll events.
    fn interest_to_epoll_events(interest: Interest) -> u32 {
        let mut events = libc::EPOLLET as u32; // Edge-triggered mode
        
        if interest.readable {
            events |= libc::EPOLLIN as u32;
        }
        
        if interest.writable {
            events |= libc::EPOLLOUT as u32;
        }
        
        if interest.error {
            events |= libc::EPOLLERR as u32 | libc::EPOLLHUP as u32;
        }
        
        events
    }
    
    /// Convert epoll events to Event.
    fn epoll_events_to_event(fd: RawFd, events: u32) -> Event {
        Event {
            fd,
            readable: (events & libc::EPOLLIN as u32) != 0,
            writable: (events & libc::EPOLLOUT as u32) != 0,
            error: (events & libc::EPOLLERR as u32) != 0,
            hangup: (events & libc::EPOLLHUP as u32) != 0,
        }
    }
}

impl Reactor for EpollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let events = Self::interest_to_epoll_events(interest);
        
        let mut event = libc::epoll_event {
            events,
            u64: fd as u64,
        };
        
        let result = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_ADD,
                fd,
                &mut event as *mut libc::epoll_event,
            )
        };
        
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        
        Ok(())
    }
    
    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let result = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_DEL,
                fd,
                std::ptr::null_mut(),
            )
        };
        
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        
        Ok(())
    }
    
    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        // Ensure event buffer has the right capacity
        let mut events = vec![
            libc::epoll_event { events: 0, u64: 0 };
            MAX_EVENTS
        ];
        
        let timeout_ms = match timeout {
            Some(duration) => duration.as_millis() as libc::c_int,
            None => -1, // Block indefinitely
        };
        
        let num_events = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                events.as_mut_ptr(),
                MAX_EVENTS as libc::c_int,
                timeout_ms,
            )
        };
        
        if num_events < 0 {
            return Err(io::Error::last_os_error());
        }
        
        let mut result = Vec::with_capacity(num_events as usize);
        
        // Use iterator pattern per Rust Book Ch.13 for better performance
        for event in events.iter().take(num_events as usize) {
            let fd = event.u64 as RawFd;
            let reactor_event = Self::epoll_events_to_event(fd, event.events);
            result.push(reactor_event);
        }
        
        Ok(result)
    }
    
    fn wake(&self) -> io::Result<()> {
        // Create a self-pipe to wake up epoll_wait
        // This is a simplified implementation
        // A complete implementation would use eventfd or a proper wakeup mechanism
        Ok(())
    }
}

impl Drop for EpollReactor {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.epoll_fd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_epoll_reactor_creation() {
        let reactor = EpollReactor::new();
        assert!(reactor.is_ok());
    }
    
    #[test]
    fn test_interest_conversion() {
        let interest = Interest::READ_WRITE;
        let events = EpollReactor::interest_to_epoll_events(interest);
        
        assert_ne!(events & libc::EPOLLIN as u32, 0);
        assert_ne!(events & libc::EPOLLOUT as u32, 0);
        assert_ne!(events & libc::EPOLLET as u32, 0);
    }
    
    #[test]
    fn test_event_conversion() {
        let fd = 5;
        let events = libc::EPOLLIN as u32 | libc::EPOLLERR as u32;
        let event = EpollReactor::epoll_events_to_event(fd, events);
        
        assert_eq!(event.fd, fd);
        assert!(event.readable);
        assert!(!event.writable);
        assert!(event.error);
        assert!(!event.hangup);
    }
}