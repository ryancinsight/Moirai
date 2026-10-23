//! Shared load algorithm; direction is a compile-time policy.

mod material;
mod metrics;
mod observation;
pub(super) mod recovery;

pub(super) use material::endpoint;
pub(super) use metrics::report;
use moirai_quic::{QuicEndpoint, QuicStream};
use std::io;
use std::net::SocketAddr;
use std::time::Instant;

// Condition 1: CLI diagnostics erase distinct TLS, transport and codec errors;
// successful message processing has no error allocation or virtual dispatch.
pub(super) type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
pub(super) struct Plan {
    pub(super) streams: u32,
    pub(super) messages: u32,
    pub(super) bytes: u32,
    pub(super) payload_bytes: u64,
    pub(super) policy: moirai_quic::CongestionPolicy,
}

impl Plan {
    pub(super) fn new(streams: u32, messages: u32, bytes: u32) -> ProbeResult<Self> {
        let payload_bytes = u64::from(streams)
            .checked_mul(u64::from(messages))
            .and_then(|count| count.checked_mul(u64::from(bytes)))
            .filter(|_| streams > 0 && messages > 0 && bytes >= 16)
            .ok_or_else(|| {
                invalid("positive counts, a sequence header and finite byte count are required")
            })?;
        Ok(Self {
            streams,
            messages,
            bytes,
            payload_bytes,
            policy: moirai_quic::CongestionPolicy::default(),
        })
    }
}

pub(super) struct Exchange {
    pub(super) connection_setup_ns: u64,
    pub(super) elapsed_ns: u64,
    pub(super) samples: Vec<u64>,
}

pub(super) async fn run<const RESPONDER: bool>(
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
    material::authorize(&connection)?;
    let connection_setup_ns = nanos(started)?;
    tracing::info!(
        phase = "authenticated",
        connection_setup_ns,
        "load probe phase"
    );
    let exchanged = Instant::now();
    let progress = observation::Progress::new(exchanged);
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
    tracing::info!(
        phase = "exchange",
        elapsed_ns = nanos(started)?,
        "load probe phase"
    );
    let exchange = futures::future::try_join_all(sessions);
    let observe = progress.observe(&connection);
    futures::pin_mut!(exchange, observe);
    let result = match futures::future::select(exchange, observe).await {
        futures::future::Either::Left((result, _)) => result?,
        futures::future::Either::Right((never, _)) => match never {},
    };
    let samples = result.into_iter().flatten().collect();
    tracing::info!(
        phase = "complete",
        elapsed_ns = nanos(started)?,
        "load probe phase"
    );
    Ok(Exchange {
        connection_setup_ns,
        elapsed_ns: nanos(exchanged)?,
        samples,
    })
}

async fn session<const RESPONDER: bool>(
    mut stream: QuicStream,
    plan: Plan,
    progress: &observation::Progress,
) -> ProbeResult<Vec<u64>> {
    let ordinal = stream.id() / 4;
    // RFC 9000 stream IDs reserve the low two bits for initiator/direction.
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

/// The header makes every (stream, sequence) unique. Body arithmetic is byte
/// arithmetic modulo 256, with a full-period increment coprime to 256.
fn fill_message(bytes: &mut [u8], ordinal: u64, sequence: u64) {
    let mut byte = ordinal.to_le_bytes()[0]
        .wrapping_mul(17)
        .wrapping_add(sequence.to_le_bytes()[0].wrapping_mul(31));
    for value in bytes.iter_mut() {
        byte = byte.wrapping_add(37);
        *value = byte;
    }
    bytes[..8].copy_from_slice(&ordinal.to_be_bytes());
    bytes[8..16].copy_from_slice(&sequence.to_be_bytes());
}

fn nanos(started: Instant) -> ProbeResult<u64> {
    Ok(u64::try_from(started.elapsed().as_nanos())?)
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
