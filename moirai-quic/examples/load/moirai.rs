//! Moirai-backed workload transport.

use super::observation::{Metrics, Progress};
use super::{Exchange, Phase, Plan, ProbeResult, emit_phase, fill_message, invalid, nanos};
use moirai_quic::{QuicEndpoint, QuicStream};
use std::net::SocketAddr;
use std::time::Instant;

pub(crate) async fn run<const RESPONDER: bool>(
    endpoint: &QuicEndpoint,
    peer: SocketAddr,
    plan: Plan,
    started: Instant,
) -> ProbeResult<Exchange> {
    let connection = if RESPONDER {
        endpoint.accept().await?
    } else {
        endpoint.connect(peer, "localhost").await?
    };
    super::material::authorize(&connection)?;
    let connection_setup_ns = nanos(started)?;
    emit_phase(Phase::Authenticated {
        connection_setup_ns,
    });
    let exchanged = Instant::now();
    let progress = Progress::new(exchanged);
    let mut sessions = Vec::new();
    sessions.try_reserve_exact(usize::try_from(plan.streams)?)?;
    for _ in 0..plan.streams {
        let stream = if RESPONDER {
            connection.accept_stream().await?
        } else {
            connection.open_stream().await?
        };
        sessions.push(session::<RESPONDER>(stream, plan, &progress));
    }
    emit_phase(Phase::Exchange {
        elapsed_ns: nanos(started)?,
    });
    let exchange = futures::future::try_join_all(sessions);
    let observe = progress.observe(moirai_async::timer::sleep, || {
        let metrics = connection.metrics();
        Metrics {
            sent_bytes: metrics.sent_bytes,
            received_bytes: metrics.received_bytes,
            lost_packets: metrics.lost_packets,
            congestion_events: metrics.congestion_events,
            congestion_window_bytes: metrics.congestion_window_bytes,
            round_trip_ns: metrics.round_trip_time.as_nanos(),
            path_mtu: metrics.path_mtu,
        }
    });
    futures::pin_mut!(exchange, observe);
    let result = match futures::future::select(exchange, observe).await {
        futures::future::Either::Left((result, _)) => result?,
        futures::future::Either::Right((never, _)) => match never {},
    };
    let samples = result.into_iter().flatten().collect();
    emit_phase(Phase::Complete {
        elapsed_ns: nanos(started)?,
    });
    Ok(Exchange {
        connection_setup_ns,
        elapsed_ns: nanos(exchanged)?,
        samples,
    })
}

async fn session<const RESPONDER: bool>(
    mut stream: QuicStream,
    plan: Plan,
    progress: &Progress,
) -> ProbeResult<Vec<u64>> {
    let ordinal = stream.id() / 4;
    if stream.id() & 0b11 != 0 || ordinal >= u64::from(plan.streams) {
        return Err(invalid("stream is outside the client's admitted bidirectional domain").into());
    }
    let mut expected = Vec::new();
    expected.try_reserve_exact(usize::try_from(plan.bytes)?)?;
    expected.resize(usize::try_from(plan.bytes)?, 0);
    let mut samples = Vec::new();
    samples.try_reserve_exact(usize::try_from(plan.messages)?)?;
    for sequence in 0..plan.messages {
        fill_message(&mut expected, ordinal, u64::from(sequence));
        let started = Instant::now();
        if !RESPONDER {
            stream.send_message(&expected).await?;
        }
        let received = stream
            .recv_message()
            .await?
            .ok_or_else(|| invalid("message precedes peer FIN"))?;
        if received != expected {
            return Err(invalid(
                "received stream/sequence/payload differs from the complete expected bytes",
            )
            .into());
        }
        if RESPONDER {
            stream.send_message(&received).await?;
        }
        let elapsed = nanos(started)?;
        samples.push(elapsed);
        progress.completed(elapsed);
    }
    if !RESPONDER {
        stream.finish()?;
    }
    if stream.recv_message().await?.is_some() {
        return Err(invalid("peer sends bytes beyond the declared message count").into());
    }
    stream.finish_and_wait().await?;
    Ok(samples)
}
