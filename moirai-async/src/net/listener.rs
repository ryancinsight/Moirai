use moirai_pal::net::AsyncTcpListener;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::net::stream::TcpStream;
use crate::net::types::{ConnectionPool, ServerStats, TcpServerConfig, TcpServerStats};

/// Native async TCP listener with connection management
pub struct TcpListener {
    inner: AsyncTcpListener,
    #[allow(dead_code)]
    config: TcpServerConfig,
    stats: Arc<ServerStats>,
    connection_pool: Arc<ConnectionPool>,
}

impl TcpListener {
    /// Bind to an address with default configuration
    pub async fn bind(addr: &str) -> io::Result<Self> {
        Self::bind_with_config(addr, TcpServerConfig::default()).await
    }

    /// Bind to an address with custom configuration
    pub async fn bind_with_config(addr: &str, config: TcpServerConfig) -> io::Result<Self> {
        use std::net::ToSocketAddrs;
        let addr_parsed = addr.to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Could not resolve address")
        })?;
        let inner = AsyncTcpListener::bind(addr_parsed).await?;
        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(config.max_connections));

        Ok(Self {
            inner,
            config,
            stats,
            connection_pool,
        })
    }

    /// Accept the next incoming connection
    pub async fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        if !self.connection_pool.try_reserve() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "Connection limit reached",
            ));
        }

        // Hold the reservation in an RAII guard so it is released on *every*
        // early exit out of this `await` point — an I/O error (the `?` below) or,
        // critically, cancellation when the caller drops the `accept` future
        // while `inner.accept()` is pending. Without the guard a cancelled accept
        // permanently leaks a reservation, and after `max_connections` such
        // cancellations the listener rejects all further connections.
        let mut reservation = ReservationGuard::new(&self.connection_pool);

        let (stream, addr) = self.inner.accept().await?;

        // Update statistics
        self.stats
            .total_connections
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .active_connections
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Convert the reservation into a tracked connection. `add_connection_reserved`
        // releases the reservation itself, so disarm the guard to avoid a double
        // release.
        let connection_id = self.connection_pool.add_connection_reserved(addr);
        reservation.disarm();

        Ok((
            TcpStream::new(
                stream,
                self.stats.clone(),
                self.connection_pool.clone(),
                Some(connection_id),
            ),
            addr,
        ))
    }

    /// Get server statistics
    pub fn stats(&self) -> TcpServerStats {
        TcpServerStats {
            total_connections: self
                .stats
                .total_connections
                .load(std::sync::atomic::Ordering::Relaxed),
            active_connections: self
                .stats
                .active_connections
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self
                .stats
                .bytes_received
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_sent: self
                .stats
                .bytes_sent
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Return the local socket address this listener is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

/// RAII guard that releases a connection-pool reservation unless disarmed.
///
/// Armed on construction; [`Self::disarm`] is called once the reservation has
/// been converted into a tracked connection. If the guard is dropped while still
/// armed — an I/O error or future cancellation between `try_reserve` and commit —
/// it calls `cancel_reservation`, so reservations are never leaked.
struct ReservationGuard<'a> {
    pool: &'a ConnectionPool,
    armed: bool,
}

impl<'a> ReservationGuard<'a> {
    fn new(pool: &'a ConnectionPool) -> Self {
        Self { pool, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.pool.cancel_reservation();
        }
    }
}
