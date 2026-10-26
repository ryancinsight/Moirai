use std::{
    any::Any,
    fmt,
    future::Future,
    io,
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{ready, Context, Poll, Waker},
};

use bytes::Bytes;
use moirai_async::sync::{mpsc, oneshot, Notify, NotifyFuture};
use pin_project_lite::pin_project;
use rustc_hash::FxHashMap;
use thiserror::Error;
use tracing::{debug_span, Instrument, Span};

use crate::{
    datagram::udp_transmit,
    mutex::Mutex,
    recv_stream::RecvStream,
    runtime::{AsyncTimer, AsyncUdpSocket, Runtime, UdpPoller},
    send_stream::SendStream,
    ConnectionEvent, Duration, Instant, VarInt,
};
use proto::{
    congestion::Controller, ConnectionError, ConnectionHandle, ConnectionStats, Dir, EndpointEvent,
    Side, StreamEvent, StreamId,
};

// The protocol assigns the bidirectional stream type index 0 and the unidirectional type 1.
// Keep notification-array indexing explicit so a future enum change cannot silently reorder it.
const fn direction_index(direction: Dir) -> usize {
    match direction {
        Dir::Bi => 0,
        Dir::Uni => 1,
    }
}

/// In-progress connection attempt future
pub struct Connecting {
    conn: Option<ConnectionRef>,
    connected: oneshot::Receiver<bool>,
    handshake_data_ready: Option<oneshot::Receiver<()>>,
}

/// Future that completes when a connection is fully established
///
/// For clients, the resulting value indicates if 0-RTT was accepted. For servers, the resulting
/// value is meaningless.
pub struct ZeroRttAccepted(oneshot::Receiver<bool>);

/// A future that drives protocol logic for a connection
///
/// This future handles the protocol logic for a single connection, routing events from the
/// `Connection` API object to the `Endpoint` task and the related stream-related interfaces.
/// It also keeps track of outstanding timeouts for the `Connection`.
///
/// If the connection encounters an error condition, this future will yield an error. It will
/// terminate (yielding `Ok(())`) if the connection was closed without error. Unlike other
/// connection-related futures, this waits for the draining period to complete to ensure that
/// packets still in flight from the peer are handled gracefully.
#[must_use = "connection drivers must be spawned for their connections to function"]
#[derive(Debug)]
struct ConnectionDriver {
    conn: ConnectionRef,
    span: Span,
}

/// A QUIC connection.
///
/// If all references to a connection (including every clone of the `Connection` handle, streams of
/// incoming streams, and the various stream types) have been dropped, then the connection will be
/// automatically closed with an `error_code` of 0 and an empty `reason`. You can also close the
/// connection explicitly by calling [`Connection::close()`].
///
/// Closing the connection immediately abandons efforts to deliver data to the peer.  Upon
/// receiving CONNECTION_CLOSE the peer *may* drop any stream data not yet delivered to the
/// application. [`Connection::close()`] describes in more detail how to gracefully close a
/// connection without losing application data.
///
/// May be cloned to obtain another handle to the same connection.
///
/// [`Connection::close()`]: Connection::close
#[derive(Debug, Clone)]
pub struct Connection(ConnectionRef);

pin_project! {
    /// Future produced by [`Connection::open_uni`]
    pub struct OpenUni<'a> {
        conn: &'a ConnectionRef,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

impl Future for OpenUni<'_> {
    type Output = Result<SendStream, ConnectionError>;
    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let (conn, id, is_0rtt) = ready!(poll_open(ctx, this.conn, this.notify, Dir::Uni))?;
        Poll::Ready(Ok(SendStream::new(conn, id, is_0rtt)))
    }
}

pin_project! {
    /// Future produced by [`Connection::open_bi`]
    pub struct OpenBi<'a> {
        conn: &'a ConnectionRef,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

impl Future for OpenBi<'_> {
    type Output = Result<(SendStream, RecvStream), ConnectionError>;
    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let (conn, id, is_0rtt) = ready!(poll_open(ctx, this.conn, this.notify, Dir::Bi))?;

        Poll::Ready(Ok((
            SendStream::new(conn.clone(), id, is_0rtt),
            RecvStream::new(conn, id, is_0rtt),
        )))
    }
}

fn poll_open<'a>(
    ctx: &mut Context<'_>,
    conn: &'a ConnectionRef,
    mut notify: Pin<&mut NotifyFuture<'a>>,
    dir: Dir,
) -> Poll<Result<(ConnectionRef, StreamId, bool), ConnectionError>> {
    let mut state = conn.state.lock("poll_open");
    if let Some(ref e) = state.error {
        return Poll::Ready(Err(e.clone()));
    } else if let Some(id) = state.inner.streams().open(dir) {
        let is_0rtt = state.inner.side().is_client() && state.inner.is_handshaking();
        drop(state); // Release the lock so clone can take it
        return Poll::Ready(Ok((conn.clone(), id, is_0rtt)));
    }
    loop {
        match notify.as_mut().poll(ctx) {
            // `state` lock ensures we didn't race with readiness
            Poll::Pending => return Poll::Pending,
            // Spurious wakeup, get a new future
            Poll::Ready(()) => {
                notify.set(conn.shared.stream_budget_available[direction_index(dir)].notified())
            }
        }
    }
}

pin_project! {
    /// Future produced by [`Connection::accept_uni`]
    pub struct AcceptUni<'a> {
        conn: &'a ConnectionRef,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

impl Future for AcceptUni<'_> {
    type Output = Result<RecvStream, ConnectionError>;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let (conn, id, is_0rtt) = ready!(poll_accept(ctx, this.conn, this.notify, Dir::Uni))?;
        Poll::Ready(Ok(RecvStream::new(conn, id, is_0rtt)))
    }
}

pin_project! {
    /// Future produced by [`Connection::accept_bi`]
    pub struct AcceptBi<'a> {
        conn: &'a ConnectionRef,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

impl Future for AcceptBi<'_> {
    type Output = Result<(SendStream, RecvStream), ConnectionError>;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let (conn, id, is_0rtt) = ready!(poll_accept(ctx, this.conn, this.notify, Dir::Bi))?;
        Poll::Ready(Ok((
            SendStream::new(conn.clone(), id, is_0rtt),
            RecvStream::new(conn, id, is_0rtt),
        )))
    }
}

fn poll_accept<'a>(
    ctx: &mut Context<'_>,
    conn: &'a ConnectionRef,
    mut notify: Pin<&mut NotifyFuture<'a>>,
    dir: Dir,
) -> Poll<Result<(ConnectionRef, StreamId, bool), ConnectionError>> {
    let mut state = conn.state.lock("poll_accept");
    // Check for incoming streams before checking `state.error` so that already-received streams,
    // which are necessarily finite, can be drained from a closed connection.
    if let Some(id) = state.inner.streams().accept(dir) {
        let is_0rtt = state.inner.is_handshaking();
        state.wake(); // To send additional stream ID credit
        drop(state); // Release the lock so clone can take it
        return Poll::Ready(Ok((conn.clone(), id, is_0rtt)));
    } else if let Some(ref e) = state.error {
        return Poll::Ready(Err(e.clone()));
    }
    loop {
        match notify.as_mut().poll(ctx) {
            // `state` lock ensures we didn't race with readiness
            Poll::Pending => return Poll::Pending,
            // Spurious wakeup, get a new future
            Poll::Ready(()) => {
                notify.set(conn.shared.stream_incoming[direction_index(dir)].notified())
            }
        }
    }
}

pin_project! {
    /// Future produced by [`Connection::read_datagram`]
    pub struct ReadDatagram<'a> {
        conn: &'a ConnectionRef,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

impl Future for ReadDatagram<'_> {
    type Output = Result<Bytes, ConnectionError>;
    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        let mut state = this.conn.state.lock("ReadDatagram::poll");
        // Check for buffered datagrams before checking `state.error` so that already-received
        // datagrams, which are necessarily finite, can be drained from a closed connection.
        if let Some(x) = state.inner.datagrams().recv() {
            return Poll::Ready(Ok(x));
        } else if let Some(ref e) = state.error {
            return Poll::Ready(Err(e.clone()));
        }
        loop {
            match this.notify.as_mut().poll(ctx) {
                // `state` lock ensures we didn't race with readiness
                Poll::Pending => return Poll::Pending,
                // Spurious wakeup, get a new future
                Poll::Ready(()) => this
                    .notify
                    .set(this.conn.shared.datagram_received.notified()),
            }
        }
    }
}

pin_project! {
    /// Future produced by [`Connection::send_datagram_wait`]
    pub struct SendDatagram<'a> {
        conn: &'a ConnectionRef,
        state: DatagramSendState,
        #[pin]
        notify: NotifyFuture<'a>,
    }
}

// Keep the completed result so a repeated poll cannot send the datagram a second time. Polling
// is only a transient state while `Future::poll` holds exclusive access to this object.
enum DatagramSendState {
    Queued(Bytes),
    Polling,
    Complete(Result<(), SendDatagramError>),
}

impl Future for SendDatagram<'_> {
    type Output = Result<(), SendDatagramError>;
    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        if let DatagramSendState::Complete(result) = this.state {
            return Poll::Ready(result.clone());
        }
        let mut state = this.conn.state.lock("SendDatagram::poll");
        if let Some(ref e) = state.error {
            let result = Err(SendDatagramError::ConnectionLost(e.clone()));
            *this.state = DatagramSendState::Complete(result.clone());
            return Poll::Ready(result);
        }
        let data = match std::mem::replace(this.state, DatagramSendState::Polling) {
            DatagramSendState::Queued(data) => data,
            DatagramSendState::Complete(result) => {
                *this.state = DatagramSendState::Complete(result.clone());
                return Poll::Ready(result);
            }
            DatagramSendState::Polling => {
                unreachable!("exclusive datagram future polled recursively")
            }
        };
        use proto::SendDatagramError::*;
        match state.inner.datagrams().send(data, false) {
            Ok(()) => {
                state.wake();
                *this.state = DatagramSendState::Complete(Ok(()));
                Poll::Ready(Ok(()))
            }
            Err(e) => {
                let error = match e {
                    Blocked(data) => {
                        *this.state = DatagramSendState::Queued(data);
                        loop {
                            match this.notify.as_mut().poll(ctx) {
                                Poll::Pending => return Poll::Pending,
                                // Spurious wakeup, get a new future
                                Poll::Ready(()) => this
                                    .notify
                                    .set(this.conn.shared.datagrams_unblocked.notified()),
                            }
                        }
                    }
                    UnsupportedByPeer => SendDatagramError::UnsupportedByPeer,
                    Disabled => SendDatagramError::Disabled,
                    TooLarge => SendDatagramError::TooLarge,
                };
                *this.state = DatagramSendState::Complete(Err(error.clone()));
                Poll::Ready(Err(error))
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ConnectionRef(Arc<ConnectionInner>);

/// Named ownership handoff from the handshake to the connection driver.
/// Keeping channels distinct prevents positional swaps between the endpoint,
/// connection and one-shot notification paths.
struct ConnectionInit {
    handle: ConnectionHandle,
    conn: proto::Connection,
    endpoint_events: mpsc::Sender<(ConnectionHandle, EndpointEvent), mpsc::Unbounded>,
    conn_events: mpsc::Receiver<ConnectionEvent, mpsc::Unbounded>,
    on_handshake_data: oneshot::Sender<()>,
    on_connected: oneshot::Sender<bool>,
    socket: Arc<dyn AsyncUdpSocket>,
    runtime: Arc<dyn Runtime>,
}

#[derive(Debug)]
pub(crate) struct ConnectionInner {
    pub(crate) state: Mutex<State>,
    pub(crate) shared: Shared,
}

#[derive(Default)]
pub(crate) struct Shared {
    /// Notified when new streams may be locally initiated due to an increase in stream ID flow
    /// control budget
    stream_budget_available: [Notify; 2],
    /// Notified when the peer has initiated a new stream
    stream_incoming: [Notify; 2],
    datagram_received: Notify,
    datagrams_unblocked: Notify,
    closed: Notify,
    /// Number of live handles that can used to initiate or handle I/O; excludes the driver
    ref_count: AtomicUsize,
}

impl fmt::Debug for Shared {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Shared")
            .field("ref_count", &self.ref_count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

pub(crate) struct State {
    pub(crate) inner: proto::Connection,
    driver: Option<Waker>,
    handle: ConnectionHandle,
    on_handshake_data: Option<oneshot::Sender<()>>,
    on_connected: Option<oneshot::Sender<bool>>,
    connected: bool,
    timer: Option<Pin<Box<dyn AsyncTimer>>>,
    timer_deadline: Option<Instant>,
    conn_events: mpsc::Receiver<ConnectionEvent, mpsc::Unbounded>,
    endpoint_events: mpsc::Sender<(ConnectionHandle, EndpointEvent), mpsc::Unbounded>,
    pub(crate) blocked_writers: FxHashMap<StreamId, Waker>,
    pub(crate) blocked_readers: FxHashMap<StreamId, Waker>,
    pub(crate) stopped: FxHashMap<StreamId, Arc<Notify>>,
    /// Always set to Some before the connection becomes drained
    pub(crate) error: Option<ConnectionError>,
    socket: Arc<dyn AsyncUdpSocket>,
    io_poller: Pin<Box<dyn UdpPoller>>,
    runtime: Arc<dyn Runtime>,
    send_buffer: Vec<u8>,
    /// We buffer a transmit when the underlying I/O would block
    buffered_transmit: Option<proto::Transmit>,
}

fn wake_stream(stream_id: StreamId, wakers: &mut FxHashMap<StreamId, Waker>) {
    if let Some(waker) = wakers.remove(&stream_id) {
        waker.wake();
    }
}

fn wake_all(wakers: &mut FxHashMap<StreamId, Waker>) {
    wakers.drain().for_each(|(_, waker)| waker.wake())
}

fn wake_stream_notify(stream_id: StreamId, wakers: &mut FxHashMap<StreamId, Arc<Notify>>) {
    if let Some(notify) = wakers.remove(&stream_id) {
        notify.notify_waiters()
    }
}

fn wake_all_notify(wakers: &mut FxHashMap<StreamId, Arc<Notify>>) {
    wakers
        .drain()
        .for_each(|(_, notify)| notify.notify_waiters())
}

/// Errors that can arise when sending a datagram
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum SendDatagramError {
    /// The peer does not support receiving datagram frames
    #[error("datagrams not supported by peer")]
    UnsupportedByPeer,
    /// Datagram support is disabled locally
    #[error("datagram support disabled")]
    Disabled,
    /// The datagram is larger than the connection can currently accommodate
    ///
    /// Indicates that the path MTU minus overhead or the limit advertised by the peer has been
    /// exceeded.
    #[error("datagram too large")]
    TooLarge,
    /// The connection was lost
    #[error("connection lost")]
    ConnectionLost(#[from] ConnectionError),
}

/// The maximum amount of datagrams which will be produced in a single `drive_transmit` call
///
/// This limits the amount of CPU resources consumed by datagram generation,
/// and allows other tasks (like receiving ACKs) to run in between.
const MAX_TRANSMIT_DATAGRAMS: usize = 20;

/// Upper bound on connection events handled under one driver-state lock.
/// Tokio's original unbounded receiver yields after 128 ready receives in a
/// fresh task poll. Explicitly retain that bound with Moirai's native receiver
/// so transmission, timers and other tasks progress with a nonempty backlog.
const MAX_CONNECTION_EVENTS_PER_POLL: usize = 128;

/// The maximum amount of datagrams that are sent in a single transmit
///
/// This can be lower than the maximum platform capabilities, to avoid excessive
/// memory allocations when calling `poll_transmit()`. Benchmarks have shown
/// that numbers around 10 are a good compromise.
const MAX_TRANSMIT_SEGMENTS: usize = 10;

mod connecting;
mod driver;
mod handle;
mod state;
