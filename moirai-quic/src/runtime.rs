//! Quinn's type-erased runtime ABI delegates to a concrete Moirai spawner.

use crate::lock_mutex;
use moirai_async::timer::Delay;
use moirai_core::executor::TaskSpawner;
use moirai_pal::net::{AsyncUdpSocket, UdpIoPoller, UdpReady};
use quinn::udp::{RecvMeta, Transmit, UdpSocketState};
use quinn::{AsyncTimer, AsyncUdpSocket as QuinnSocket, Runtime, UdpPoller};
use std::fmt;
use std::future::Future;
use std::io::{self, IoSliceMut};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;
use tracing::instrument::WithSubscriber;
mod health;
pub(super) use health::RuntimeHealth;

pub(super) struct MoiraiRuntime<E> {
    pub(super) executor: Arc<E>,
    pub(super) health: Arc<RuntimeHealth>,
}

impl<E> fmt::Debug for MoiraiRuntime<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoiraiRuntime")
            .field("executor", &std::any::type_name::<E>())
            .finish()
    }
}

impl<E: TaskSpawner + Send + Sync + 'static> Runtime for MoiraiRuntime<E> {
    // Condition 1: Quinn requires a type-erased runtime/timer ABI. The concrete
    // Moirai spawner remains monomorphized; no second scheduler is constructed.
    fn new_timer(&self, deadline: Instant) -> Pin<Box<dyn AsyncTimer>> {
        Box::pin(Timer(Delay::until(deadline)))
    }

    // Condition 1: Quinn supplies an erased protocol driver through its ABI.
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) {
        if let Err(error) = self.executor.spawn_async(future.with_current_subscriber()) {
            self.health.fail(error);
        }
    }

    // Condition 1: Quinn stores a runtime-erased native socket at its boundary.
    fn wrap_udp_socket(&self, socket: std::net::UdpSocket) -> io::Result<Arc<dyn QuinnSocket>> {
        let state = UdpSocketState::new((&socket).into())?;
        let socket = AsyncUdpSocket::from_std(socket)?;
        let receive = Mutex::new(socket.recv_poller());
        Ok(Arc::new(Socket {
            receive,
            socket,
            state,
            writable: Arc::new(AtomicBool::new(false)),
        }))
    }
}

struct Timer(Delay);

impl fmt::Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Timer").field(&self.0.deadline()).finish()
    }
}

impl AsyncTimer for Timer {
    fn reset(self: Pin<&mut Self>, deadline: Instant) {
        self.get_mut().0 = Delay::until(deadline);
    }

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        Pin::new(&mut self.get_mut().0).poll(cx)
    }
}

struct Socket {
    // The poller owns its descriptor and cancellation token independently.
    receive: Mutex<UdpIoPoller>,
    socket: AsyncUdpSocket,
    state: UdpSocketState,
    writable: Arc<AtomicBool>,
}

impl fmt::Debug for Socket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Socket")
            .field("address", &self.socket.local_addr())
            .finish_non_exhaustive()
    }
}

impl QuinnSocket for Socket {
    // Condition 1: Quinn's ABI erases independently cancellable write waiters.
    fn create_io_poller(self: Arc<Self>) -> Pin<Box<dyn UdpPoller>> {
        Box::pin(WritePoller {
            ready: self.socket.writable(),
            writable: Arc::clone(&self.writable),
        })
    }

    fn try_send(&self, transmit: &Transmit<'_>) -> io::Result<()> {
        // Quinn treats non-backpressure UDP errors as packet loss and logs
        // their OS context; reliable completion still requires acknowledgements.
        // Keeping its UDP policy also preserves MTU probing on EMSGSIZE.
        let result = self.state.send(self.socket.as_std().into(), transmit);
        if result
            .as_ref()
            .is_err_and(|error| error.kind() == io::ErrorKind::WouldBlock)
        {
            self.writable.store(false, Ordering::Release);
        }
        result
    }

    fn poll_recv(
        &self,
        cx: &mut Context<'_>,
        bufs: &mut [IoSliceMut<'_>],
        meta: &mut [RecvMeta],
    ) -> Poll<io::Result<usize>> {
        lock_mutex(&self.receive).poll_io(cx, || {
            self.state.recv(self.socket.as_std().into(), bufs, meta)
        })
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }
    fn max_transmit_segments(&self) -> usize {
        self.state.max_gso_segments()
    }
    fn max_receive_segments(&self) -> usize {
        self.state.gro_segments()
    }
    fn may_fragment(&self) -> bool {
        self.state.may_fragment()
    }
}

struct WritePoller {
    ready: UdpReady,
    writable: Arc<AtomicBool>,
}

impl fmt::Debug for WritePoller {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WritePoller").finish_non_exhaustive()
    }
}

impl UdpPoller for WritePoller {
    fn poll_writable(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        // Quinn polls before each datagram, not only after WouldBlock. Retain
        // the real readiness hint until a send exhausts it, avoiding one native
        // register/cancel cycle per packet. Every send still attempts kernel I/O.
        if this.writable.load(Ordering::Acquire) {
            return Poll::Ready(Ok(()));
        }
        let result = this.ready.poll_ready(cx);
        if matches!(result, Poll::Ready(Ok(()))) {
            this.writable.store(true, Ordering::Release);
        }
        result
    }
}

#[cfg(test)]
mod tests;
