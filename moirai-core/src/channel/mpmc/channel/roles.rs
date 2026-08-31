use super::MpmcChannel;
use crate::channel::error::{Channel, Result};
use crate::channel::roles::{Consumer, Producer};

impl<T: Send> Producer<T> for MpmcChannel<T> {
    #[inline]
    fn send(&self, value: T) -> Result<()> {
        Channel::send(self, value)
    }

    #[inline]
    fn try_send(&self, value: T) -> Result<()> {
        Channel::try_send(self, value)
    }

    #[inline]
    fn is_full(&self) -> bool {
        Channel::is_full(self)
    }

    #[inline]
    fn capacity(&self) -> Option<usize> {
        Channel::capacity(self)
    }
}

impl<T: Send> Consumer<T> for MpmcChannel<T> {
    #[inline]
    fn recv(&self) -> Result<T> {
        Channel::recv(self)
    }

    #[inline]
    fn try_recv(&self) -> Result<T> {
        Channel::try_recv(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Channel::is_empty(self)
    }
}
