//! Integer measurements retain the source counts and rounding law.

use super::{Exchange, Plan, ProbeResult, invalid};
use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Report {
    schema: u8,
    responder: bool,
    streams: u32,
    messages_per_stream: u32,
    frame_bytes: u32,
    verified_payload_bytes_per_direction: u64,
    connection_setup_ns: u64,
    exchange_ns: u64,
    goodput_bits_per_second: u128,
    p50_ns: u64,
    p95_ns: u64,
    samples_ns: Vec<u64>,
}

pub(crate) fn report<const RESPONDER: bool>(
    plan: Plan,
    mut result: Exchange,
) -> ProbeResult<Report> {
    let count = u64::from(plan.streams) * u64::from(plan.messages);
    if u64::try_from(result.samples.len())? != count || result.elapsed_ns == 0 {
        return Err(
            invalid("sample count or elapsed duration violates the exchange contract").into(),
        );
    }
    result.samples.sort_unstable();
    // Widening is required: a u64 byte count times 8*10^9 fits in u128;
    // the exact integer quotient defines the documented rate's rounding.
    let goodput_bits_per_second =
        u128::from(plan.payload_bytes) * 8 * 1_000_000_000 / u128::from(result.elapsed_ns);
    Ok(Report {
        schema: 1,
        responder: RESPONDER,
        streams: plan.streams,
        messages_per_stream: plan.messages,
        frame_bytes: plan.bytes,
        verified_payload_bytes_per_direction: plan.payload_bytes,
        connection_setup_ns: result.connection_setup_ns,
        exchange_ns: result.elapsed_ns,
        goodput_bits_per_second,
        p50_ns: percentile(&result.samples, 50),
        p95_ns: percentile(&result.samples, 95),
        samples_ns: result.samples,
    })
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    // Quotient/remainder form avoids multiplying the complete length by 100.
    let rank = sorted.len() / 100 * percent + (sorted.len() % 100 * percent).div_ceil(100);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_preserves_exact_count_rate_and_nearest_rank_samples() {
        let subscriber = tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish();
        let _trace = tracing::subscriber::set_default(subscriber);
        let arrange = tracing::info_span!("arrange").entered();
        let plan = Plan::new(4, 5, 24).expect("four streams, five complete headers");
        let samples: Vec<_> = (0..20).rev().map(|index| 3 * index + 2).collect();
        drop(arrange);
        let act = tracing::info_span!("act").entered();
        let measured = report::<false>(
            plan,
            Exchange {
                connection_setup_ns: 19,
                elapsed_ns: 100_000_000,
                samples,
            },
        )
        .expect("computed report");
        drop(act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(measured.verified_payload_bytes_per_direction, 480);
        assert_eq!(measured.goodput_bits_per_second, 38_400);
        assert_eq!(measured.p50_ns, 29);
        assert_eq!(measured.p95_ns, 56);
        assert_eq!(
            measured.samples_ns,
            (0..20).map(|index| 3 * index + 2).collect::<Vec<_>>()
        );
        assert_eq!(percentile(&[37], 50), 37);
        assert_eq!(percentile(&[37], 95), 37);
        for (streams, messages, bytes) in [
            (0, 5, 24),
            (4, 0, 24),
            (4, 5, 15),
            (u32::MAX, u32::MAX, u32::MAX),
        ] {
            assert_eq!(
                Plan::new(streams, messages, bytes)
                    .err()
                    .expect("rejected plan")
                    .downcast_ref::<std::io::Error>()
                    .expect("plan error")
                    .kind(),
                std::io::ErrorKind::InvalidData
            );
        }
    }
}
