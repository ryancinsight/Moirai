//! Low-frequency recovery and value-verified progress observations.

use moirai_quic::QuicConnection;
use std::convert::Infallible;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(super) struct Progress {
    started: Instant,
    completed: Mutex<Vec<u64>>,
}

impl Progress {
    pub(super) fn new(started: Instant) -> Self {
        Self {
            started,
            completed: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn completed(&self, elapsed: u64) {
        self.completed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(elapsed);
    }

    pub(super) async fn observe(&self, connection: &QuicConnection) -> Infallible {
        loop {
            moirai_async::timer::sleep(Duration::from_secs(1)).await;
            let metrics = connection.metrics();
            let mut samples = self
                .completed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            samples.sort_unstable();
            let p95 = samples
                .get((samples.len() * 95).div_ceil(100).saturating_sub(1))
                .copied();
            tracing::info!(
                observed_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).expect("probe clock follows epoch").as_nanos(),
                exchange_elapsed_ns = self.started.elapsed().as_nanos(),
                completed_messages = samples.len(),
                completed_cycle_p95_ns = ?p95,
                sent_bytes = metrics.sent_bytes,
                received_bytes = metrics.received_bytes,
                lost_packets = metrics.lost_packets,
                congestion_events = metrics.congestion_events,
                congestion_window_bytes = metrics.congestion_window_bytes,
                round_trip_ns = metrics.round_trip_time.as_nanos(),
                path_mtu = metrics.path_mtu,
                "verified progress and recovery snapshot"
            );
        }
    }
}
