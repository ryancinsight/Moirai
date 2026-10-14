use std::sync::Arc;

/// Bounded selection of the pinned protocol engine's congestion controllers.
///
/// Selection occurs once at endpoint construction. All policies retain the
/// same authentication, admission, framing, flow-control and deadline contracts.
/// CUBIC remains the default; changing policy requires deployment measurements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CongestionPolicy {
    /// Quinn's stock CUBIC controller and configuration (the existing default).
    #[default]
    Cubic,
    /// Quinn's stock NewReno controller and configuration.
    NewReno,
    /// Quinn's explicitly experimental BBR implementation, for measured opt-in.
    ///
    /// Upstream does not establish this controller as a production replacement.
    /// Completing a workload alone proves neither fairness nor bounded latency.
    ExperimentalBbr,
}

impl CongestionPolicy {
    pub(super) fn configure(self, transport: &mut quinn::TransportConfig) {
        // Condition 1: Quinn owns the erased controller-factory ABI. Policy is
        // selected at construction; Moirai does not dispatch per packet here.
        match self {
            Self::Cubic => transport
                .congestion_controller_factory(Arc::new(quinn::congestion::CubicConfig::default())),
            Self::NewReno => transport.congestion_controller_factory(Arc::new(
                quinn::congestion::NewRenoConfig::default(),
            )),
            Self::ExperimentalBbr => transport
                .congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default())),
        };
    }
}
