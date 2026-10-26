//! Internal UDP datagram conversion and endpoint work bounds.

use crate::Duration;

pub(crate) fn udp_transmit<'a>(t: &proto::Transmit, buffer: &'a [u8]) -> udp::Transmit<'a> {
    udp::Transmit {
        destination: t.destination,
        ecn: t.ecn.map(udp_ecn),
        contents: buffer,
        segment_size: t.segment_size,
        src_ip: t.src_ip,
    }
}

fn udp_ecn(ecn: proto::EcnCodepoint) -> udp::EcnCodepoint {
    match ecn {
        proto::EcnCodepoint::Ect0 => udp::EcnCodepoint::Ect0,
        proto::EcnCodepoint::Ect1 => udp::EcnCodepoint::Ect1,
        proto::EcnCodepoint::Ce => udp::EcnCodepoint::Ce,
    }
}

/// Maximum number of datagrams processed in send/recv calls before moving on to other work.
///
/// This keeps a busy socket from starving the runtime. The bound was selected by upstream
/// throughput measurements.
pub(crate) const IO_LOOP_BOUND: usize = 160;

/// Maximum time spent in recvmsg calls per endpoint iteration.
///
/// A 50-microsecond receive bound plus the send bound limits one endpoint iteration to roughly
/// 100 microseconds. A single 32-message recvmmsg batch can itself take about 30 microseconds.
pub(crate) const RECV_TIME_BOUND: Duration = Duration::from_micros(50);
