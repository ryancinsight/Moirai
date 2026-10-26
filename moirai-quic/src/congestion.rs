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
    // Condition 1: Quinn owns the erased controller-factory ABI. Selection
    // happens at construction; Moirai adds no per-packet virtual dispatch.
    pub(super) fn factory(
        self,
    ) -> Arc<dyn moirai_quinn::congestion::ControllerFactory + Send + Sync> {
        match self {
            Self::Cubic => Arc::new(moirai_quinn::congestion::CubicConfig::default()),
            Self::NewReno => Arc::new(moirai_quinn::congestion::NewRenoConfig::default()),
            Self::ExperimentalBbr => Arc::new(moirai_quinn::congestion::BbrConfig::default()),
        }
    }
}
