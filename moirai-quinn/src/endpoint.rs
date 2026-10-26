use std::{
    collections::VecDeque,
    fmt,
    future::Future,
    io,
    io::IoSliceMut,
    mem,
    net::{SocketAddr, SocketAddrV6},
    pin::Pin,
    str,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
    task::{Context, Poll, Waker},
};

#[cfg(all(not(wasm_browser), any(feature = "aws-lc-rs", feature = "ring")))]
use crate::runtime::default_runtime;
use crate::{
    datagram::{udp_transmit, IO_LOOP_BOUND, RECV_TIME_BOUND},
    runtime::{AsyncUdpSocket, Runtime},
    Instant,
};
use bytes::{Bytes, BytesMut};
use moirai_async::sync::{mpsc, Notify, NotifyFuture};
use pin_project_lite::pin_project;
use proto::{
    self as proto, ClientConfig, ConnectError, ConnectionError, ConnectionHandle, DatagramEvent,
    EndpointEvent, ServerConfig,
};
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(all(not(wasm_browser), any(feature = "aws-lc-rs", feature = "ring"),))]
use socket2::{Domain, Protocol, Socket, Type};
use tracing::{Instrument, Span};
use udp::{RecvMeta, BATCH_SIZE};

use crate::{
    connection::Connecting, incoming::Incoming, work_limiter::WorkLimiter, ConnectionEvent,
    EndpointConfig, VarInt,
};

/// A QUIC endpoint.
///
/// An endpoint corresponds to a single UDP socket, may host many connections, and may act as both
/// client and server for different connections.
///
/// May be cloned to obtain another handle to the same endpoint.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub(crate) inner: EndpointRef,
    pub(crate) default_client_config: Option<ClientConfig>,
    runtime: Arc<dyn Runtime>,
}

/// Statistics on [Endpoint] activity
#[non_exhaustive]
#[derive(Debug, Default, Copy, Clone)]
pub struct EndpointStats {
    /// Cummulative number of Quic handshakes accepted by this [Endpoint]
    pub accepted_handshakes: u64,
    /// Cummulative number of Quic handshakees sent from this [Endpoint]
    pub outgoing_handshakes: u64,
    /// Cummulative number of Quic handshakes refused on this [Endpoint]
    pub refused_handshakes: u64,
    /// Cummulative number of Quic handshakes ignored on this [Endpoint]
    pub ignored_handshakes: u64,
}

/// A future that drives IO on an endpoint
///
/// This task functions as the switch point between the UDP socket object and the
/// `Endpoint` responsible for routing datagrams to their owning `Connection`.
/// In order to do so, it also facilitates the exchange of different types of events
/// flowing between the `Endpoint` and the tasks managing `Connection`s. As such,
/// running this task is necessary to keep the endpoint's connections running.
///
/// `EndpointDriver` futures terminate when all clones of the `Endpoint` have been dropped, or when
/// an I/O error occurs.
#[must_use = "endpoint drivers must be spawned for I/O to occur"]
#[derive(Debug)]
pub(crate) struct EndpointDriver(pub(crate) EndpointRef);

#[derive(Debug)]
pub(crate) struct EndpointInner {
    pub(crate) state: Mutex<State>,
    pub(crate) shared: Shared,
}

impl EndpointInner {
    // Protocol state may be partially updated if a user-provided runtime or socket panics while
    // the lock is held. A poisoned state cannot be resumed as a live QUIC endpoint. Fallible
    // callers propagate closure; infallible callers fail closed with one explicit diagnostic.
    fn lock_state(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(state) => state,
            Err(_) => panic!("QUIC endpoint state poisoned; cannot resume protocol processing"),
        }
    }

    pub(crate) fn accept(
        &self,
        incoming: proto::Incoming,
        server_config: Option<Arc<ServerConfig>>,
    ) -> Result<Connecting, ConnectionError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ConnectionError::LocallyClosed)?;
        let mut response_buffer = Vec::new();
        let now = state.runtime.now();
        match state
            .inner
            .accept(incoming, now, &mut response_buffer, server_config)
        {
            Ok((handle, conn)) => {
                state.stats.accepted_handshakes += 1;
                let socket = state.socket.clone();
                let runtime = state.runtime.clone();
                Ok(state
                    .recv_state
                    .connections
                    .insert(handle, conn, socket, runtime))
            }
            Err(error) => {
                if let Some(transmit) = error.response {
                    respond(transmit, &response_buffer, &*state.socket);
                }
                Err(error.cause)
            }
        }
    }

    pub(crate) fn refuse(&self, incoming: proto::Incoming) {
        let mut state = self.lock_state();
        state.stats.refused_handshakes += 1;
        let mut response_buffer = Vec::new();
        let transmit = state.inner.refuse(incoming, &mut response_buffer);
        respond(transmit, &response_buffer, &*state.socket);
    }

    pub(crate) fn retry(&self, incoming: proto::Incoming) -> Result<(), proto::RetryError> {
        let mut state = self.lock_state();
        let mut response_buffer = Vec::new();
        let transmit = state.inner.retry(incoming, &mut response_buffer)?;
        respond(transmit, &response_buffer, &*state.socket);
        Ok(())
    }

    pub(crate) fn ignore(&self, incoming: proto::Incoming) {
        let mut state = self.lock_state();
        state.stats.ignored_handshakes += 1;
        state.inner.ignore(incoming);
    }
}

#[derive(Debug)]
pub(crate) struct State {
    socket: Arc<dyn AsyncUdpSocket>,
    /// Earlier sockets remain readable until every connection that could still receive on them
    /// migrates to a newer socket or drains.
    retiring_sockets: Vec<RetiringSocket>,
    retired_poll_cursor: usize,
    retired_poll_remaining: usize,
    inner: proto::Endpoint,
    recv_state: RecvState,
    driver: Option<Waker>,
    ipv6: bool,
    events: mpsc::Receiver<(ConnectionHandle, EndpointEvent), mpsc::Unbounded>,
    driver_lost: bool,
    runtime: Arc<dyn Runtime>,
    stats: EndpointStats,
}

#[derive(Debug)]
struct RetiringSocket {
    socket: Arc<dyn AsyncUdpSocket>,
    pending: FxHashSet<ConnectionHandle>,
}

fn observe_connection_on(
    retiring: &mut [RetiringSocket],
    generation: usize,
    handle: ConnectionHandle,
) {
    for older in retiring.iter_mut().take(generation) {
        older.pending.remove(&handle);
    }
}

pub(crate) struct Shared {
    incoming: Notify,
    idle: Notify,
    /// Number of live handles that can be used to initiate or handle I/O; excludes the driver
    ref_count: AtomicUsize,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Shared")
            .field("ref_count", &self.ref_count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

fn respond(transmit: proto::Transmit, response_buffer: &[u8], socket: &dyn AsyncUdpSocket) {
    // Send if there's kernel buffer space; otherwise, drop it
    //
    // As an endpoint-generated packet, we know this is an
    // immediate, stateless response to an unconnected peer,
    // one of:
    //
    // - A version negotiation response due to an unknown version
    // - A `CLOSE` due to a malformed or unwanted connection attempt
    // - A stateless reset due to an unrecognized connection
    // - A `Retry` packet due to a connection attempt when
    //   `use_retry` is set
    //
    // In each case, a well-behaved peer can be trusted to retry a
    // few times, which is guaranteed to produce the same response
    // from us. Repeated failures might at worst cause a peer's new
    // connection attempt to time out, which is acceptable if we're
    // under such heavy load that there's never room for this code
    // to transmit. This is morally equivalent to the packet getting
    // lost due to congestion further along the link, which
    // similarly relies on peer retries for recovery.
    _ = socket.try_send(&udp_transmit(&transmit, &response_buffer[..transmit.size]));
}

#[inline]
fn proto_ecn(ecn: udp::EcnCodepoint) -> proto::EcnCodepoint {
    match ecn {
        udp::EcnCodepoint::Ect0 => proto::EcnCodepoint::Ect0,
        udp::EcnCodepoint::Ect1 => proto::EcnCodepoint::Ect1,
        udp::EcnCodepoint::Ce => proto::EcnCodepoint::Ce,
    }
}

#[derive(Debug)]
struct ConnectionSet {
    /// Senders for communicating with the endpoint's connections
    senders: FxHashMap<ConnectionHandle, mpsc::Sender<ConnectionEvent, mpsc::Unbounded>>,
    /// Stored to give out clones to new ConnectionInners
    sender: mpsc::Sender<(ConnectionHandle, EndpointEvent), mpsc::Unbounded>,
    /// Set if the endpoint has been manually closed
    close: Option<(VarInt, Bytes)>,
}

impl ConnectionSet {
    /// A connection can be drained between protocol routing and delivery.
    /// Returning false leaves a stale event unobserved instead of panicking
    /// or reviving the removed connection.
    fn dispatch_if_live(&mut self, handle: ConnectionHandle, event: ConnectionEvent) -> bool {
        self.senders
            .get_mut(&handle)
            .is_some_and(|sender| sender.enqueue(event).is_ok())
    }

    fn insert(
        &mut self,
        handle: ConnectionHandle,
        conn: proto::Connection,
        socket: Arc<dyn AsyncUdpSocket>,
        runtime: Arc<dyn Runtime>,
    ) -> Connecting {
        let (send, recv) = mpsc::unbounded_channel();
        if let Some((error_code, ref reason)) = self.close {
            send.enqueue(ConnectionEvent::Close {
                error_code,
                reason: reason.clone(),
            })
            .expect("a new receiver remains alive until the connection is constructed");
        }
        self.senders.insert(handle, send);
        Connecting::new(handle, conn, self.sender.clone(), recv, socket, runtime)
    }

    fn is_empty(&self) -> bool {
        self.senders.is_empty()
    }
}

fn ensure_ipv6(x: SocketAddr) -> SocketAddrV6 {
    match x {
        SocketAddr::V6(x) => x,
        SocketAddr::V4(x) => SocketAddrV6::new(x.ip().to_ipv6_mapped(), x.port(), 0, 0),
    }
}

pin_project! {
    /// Future produced by [`Endpoint::accept`]
    pub struct Accept<'a> {
        endpoint: &'a Endpoint,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

#[derive(Debug)]
pub(crate) struct EndpointRef(Arc<EndpointInner>);

impl EndpointRef {
    pub(crate) fn new(
        socket: Arc<dyn AsyncUdpSocket>,
        inner: proto::Endpoint,
        ipv6: bool,
        runtime: Arc<dyn Runtime>,
    ) -> Self {
        let (sender, events) = mpsc::unbounded_channel();
        let recv_state = RecvState::new(sender, socket.max_receive_segments(), &inner);
        Self(Arc::new(EndpointInner {
            shared: Shared {
                incoming: Notify::new(),
                idle: Notify::new(),
                ref_count: AtomicUsize::new(0),
            },
            state: Mutex::new(State {
                socket,
                retiring_sockets: Vec::new(),
                retired_poll_cursor: 0,
                retired_poll_remaining: 0,
                inner,
                ipv6,
                events,
                driver: None,
                driver_lost: false,
                recv_state,
                runtime,
                stats: EndpointStats::default(),
            }),
        }))
    }
}

impl Clone for EndpointRef {
    fn clone(&self) -> Self {
        self.0.shared.ref_count.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}

impl Drop for EndpointRef {
    fn drop(&mut self) {
        if self.shared.ref_count.fetch_sub(1, Ordering::Relaxed) > 1 {
            return;
        }

        let poisoned = self.0.state.is_poisoned();
        let mut endpoint = match self.0.state.lock() {
            Ok(endpoint) => endpoint,
            // The last handle must still release channels and wake the driver during unwind.
            // It does not resume protocol operations on a poisoned state.
            Err(poisoned) => poisoned.into_inner(),
        };
        // If the driver is about to be on its own, ensure it can shut down if the last
        // connection is gone.
        let task = endpoint.driver.take();
        let cleanup = poisoned.then(|| {
            endpoint.driver_lost = true;
            (
                mem::take(&mut endpoint.recv_state.connections.senders),
                mem::take(&mut endpoint.retiring_sockets),
            )
        });
        drop(endpoint);
        if poisoned {
            self.shared.incoming.notify_waiters();
            self.shared.idle.notify_waiters();
        }
        drop(cleanup);
        if let Some(task) = task {
            task.wake();
        }
    }
}

impl std::ops::Deref for EndpointRef {
    type Target = EndpointInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
#[path = "endpoint/event_dispatch_tests.rs"]
mod event_dispatch_tests;

/// State directly involved in handling incoming packets
struct RecvState {
    incoming: VecDeque<proto::Incoming>,
    connections: ConnectionSet,
    recv_buf: Box<[u8]>,
    recv_limiter: WorkLimiter,
}

#[derive(Default)]
struct PollProgress {
    /// Whether datagram handling was interrupted early by the work limiter for fairness
    keep_going: bool,
}

mod driver;
mod handle;
mod receive;
