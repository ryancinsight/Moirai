//! Runtime-neutral workload inputs, payloads and measurements.

use std::io;
use std::time::Instant;

// CLI diagnostics erase distinct transport and codec errors. The independent
// transport leaves retain typed errors until this outer presentation boundary.
pub(crate) type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
pub(crate) struct Plan {
    pub(crate) streams: u32,
    pub(crate) messages: u32,
    pub(crate) bytes: u32,
    pub(crate) payload_bytes: u64,
    pub(crate) policy: moirai_quic::CongestionPolicy,
}

impl Plan {
    pub(crate) fn new(streams: u32, messages: u32, bytes: u32) -> ProbeResult<Self> {
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

pub(crate) struct Exchange {
    pub(crate) connection_setup_ns: u64,
    pub(crate) elapsed_ns: u64,
    pub(crate) samples: Vec<u64>,
}

pub(crate) const PHASE_TARGET: &str = "quic_load::load";

pub(crate) enum Phase {
    Authenticated { connection_setup_ns: u64 },
    Exchange { elapsed_ns: u64 },
    Complete { elapsed_ns: u64 },
    Drain,
}

pub(crate) fn emit_phase(phase: Phase) {
    match phase {
        Phase::Authenticated {
            connection_setup_ns,
        } => tracing::info!(
            target: PHASE_TARGET,
            phase = "authenticated",
            connection_setup_ns,
            "load probe phase"
        ),
        Phase::Exchange { elapsed_ns } => tracing::info!(
            target: PHASE_TARGET,
            phase = "exchange",
            elapsed_ns,
            "load probe phase"
        ),
        Phase::Complete { elapsed_ns } => tracing::info!(
            target: PHASE_TARGET,
            phase = "complete",
            elapsed_ns,
            "load probe phase"
        ),
        Phase::Drain => tracing::info!(
            target: PHASE_TARGET,
            phase = "drain",
            "load probe phase"
        ),
    }
}

/// The header makes every (stream, sequence) unique. Body arithmetic is byte
/// arithmetic modulo 256, with a full-period increment coprime to 256.
pub(crate) fn fill_message(bytes: &mut [u8], ordinal: u64, sequence: u64) {
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

pub(crate) fn nanos(started: Instant) -> Result<u64, std::num::TryFromIntError> {
    u64::try_from(started.elapsed().as_nanos())
}

pub(crate) fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
