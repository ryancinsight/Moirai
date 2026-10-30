//! Raw Quinn connection and stream workload orchestration.

use super::super::observation::{Metrics, Progress};
use super::super::{Exchange, Phase, Plan, emit_phase, fill_message, nanos};
use super::error::{Error, Result};
use super::framing::Stream;
use super::material::{OPERATION_TIMEOUT, authorize};
use std::net::SocketAddr;
use std::time::Instant;

pub(crate) async fn run<const RESPONDER: bool>(
    endpoint: &quinn::Endpoint,
    peer: SocketAddr,
    plan: Plan,
    started: Instant,
) -> Result<Exchange> {
    let connection = if RESPONDER {
        accept(endpoint).await?
    } else {
        connect(endpoint, peer, "localhost").await?
    };
    authorize(&connection)?;
    let connection_setup_ns = nanos(started).map_err(Error::Numeric)?;
    emit_phase(Phase::Authenticated {
        connection_setup_ns,
    });
    let exchanged = Instant::now();
    let progress = Progress::new(exchanged);
    let mut sessions = Vec::new();
    sessions
        .try_reserve_exact(usize::try_from(plan.streams).map_err(Error::Numeric)?)
        .map_err(Error::Allocation)?;
    for _ in 0..plan.streams {
        let (send, recv) = if RESPONDER {
            tokio::time::timeout(OPERATION_TIMEOUT, connection.accept_bi())
                .await
                .map_err(|_| Error::TimedOut("accept stream"))?
                .map_err(Error::Connection)?
        } else {
            tokio::time::timeout(OPERATION_TIMEOUT, connection.open_bi())
                .await
                .map_err(|_| Error::TimedOut("open stream"))?
                .map_err(Error::Connection)?
        };
        sessions.push(session::<RESPONDER>(
            Stream::new(send, recv, plan.bytes),
            plan,
            &progress,
        ));
    }
    emit_phase(Phase::Exchange {
        elapsed_ns: nanos(started).map_err(Error::Numeric)?,
    });
    let exchange = futures::future::try_join_all(sessions);
    let observe = progress.observe(tokio::time::sleep, || {
        let stats = connection.stats();
        Metrics {
            sent_bytes: stats.udp_tx.bytes,
            received_bytes: stats.udp_rx.bytes,
            lost_packets: stats.path.lost_packets,
            congestion_events: stats.path.congestion_events,
            congestion_window_bytes: stats.path.cwnd,
            round_trip_ns: stats.path.rtt.as_nanos(),
            path_mtu: stats.path.current_mtu,
        }
    });
    futures::pin_mut!(exchange, observe);
    let result = match futures::future::select(exchange, observe).await {
        futures::future::Either::Left((result, _)) => result?,
        futures::future::Either::Right((never, _)) => match never {},
    };
    emit_phase(Phase::Complete {
        elapsed_ns: nanos(started).map_err(Error::Numeric)?,
    });
    Ok(Exchange {
        connection_setup_ns,
        elapsed_ns: nanos(exchanged).map_err(Error::Numeric)?,
        samples: result.into_iter().flatten().collect(),
    })
}

pub(super) async fn connect(
    endpoint: &quinn::Endpoint,
    peer: SocketAddr,
    server_name: &str,
) -> Result<quinn::Connection> {
    let connecting = endpoint
        .connect(peer, server_name)
        .map_err(Error::Connect)?;
    tokio::time::timeout(OPERATION_TIMEOUT, connecting)
        .await
        .map_err(|_| Error::TimedOut("handshake"))?
        .map_err(Error::Connection)
}

pub(super) async fn accept(endpoint: &quinn::Endpoint) -> Result<quinn::Connection> {
    let operation = async {
        let incoming = endpoint.accept().await.ok_or(Error::Closed)?;
        incoming
            .accept()
            .map_err(Error::Connection)?
            .await
            .map_err(Error::Connection)
    };
    tokio::time::timeout(OPERATION_TIMEOUT, operation)
        .await
        .map_err(|_| Error::TimedOut("accept"))?
}

async fn session<const RESPONDER: bool>(
    mut stream: Stream,
    plan: Plan,
    progress: &Progress,
) -> Result<Vec<u64>> {
    let ordinal = stream.id() / 4;
    if stream.id() & 0b11 != 0 || ordinal >= u64::from(plan.streams) {
        return Err(Error::StreamDomain);
    }
    let capacity = usize::try_from(plan.bytes).map_err(Error::Numeric)?;
    let mut expected = Vec::new();
    expected
        .try_reserve_exact(capacity)
        .map_err(Error::Allocation)?;
    expected.resize(capacity, 0);
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(usize::try_from(plan.messages).map_err(Error::Numeric)?)
        .map_err(Error::Allocation)?;
    for sequence in 0..plan.messages {
        fill_message(&mut expected, ordinal, u64::from(sequence));
        let started = Instant::now();
        if !RESPONDER {
            stream.send_message(&expected).await?;
        }
        let received = stream.recv_message().await?.ok_or(Error::Closed)?;
        verify_payload(&received, &expected)?;
        if RESPONDER {
            stream.send_message(&received).await?;
        }
        let elapsed = nanos(started).map_err(Error::Numeric)?;
        samples.push(elapsed);
        progress.completed(elapsed);
    }
    if !RESPONDER {
        stream.finish()?;
    }
    reject_extra(stream.recv_message().await?)?;
    stream.finish_and_wait().await?;
    Ok(samples)
}

pub(super) fn verify_payload(received: &[u8], expected: &[u8]) -> Result<()> {
    if received == expected {
        Ok(())
    } else {
        Err(Error::PayloadMismatch)
    }
}

pub(super) fn reject_extra(message: Option<Vec<u8>>) -> Result<()> {
    if message.is_none() {
        Ok(())
    } else {
        Err(Error::ExtraData)
    }
}
