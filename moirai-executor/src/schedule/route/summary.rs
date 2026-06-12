//! Route sequence summaries.

use super::{AcceleratorKind, AsyncLaneId, SchedulerRoute};

/// Value summary of a route sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RouteSummary {
    /// Number of local thread routes.
    pub thread_routes: usize,
    /// Number of process routes.
    pub process_routes: usize,
    /// Number of server routes.
    pub server_routes: usize,
    /// Number of accelerator metadata routes.
    pub accelerator_routes: usize,
    /// Number of CPU accelerator metadata routes.
    pub cpu_routes: usize,
    /// Number of GPU accelerator metadata routes.
    pub gpu_routes: usize,
    /// Number of TPU accelerator metadata routes.
    pub tpu_routes: usize,
    /// Number of NPU accelerator metadata routes.
    pub npu_routes: usize,
    /// Number of routes with an async lane.
    pub async_lane_routes: usize,
    /// Deterministic checksum over route placements.
    pub checksum: usize,
}

impl RouteSummary {
    /// Return the total number of summarized route decisions.
    #[inline]
    pub const fn total_routes(self) -> usize {
        self.thread_routes + self.process_routes + self.server_routes + self.accelerator_routes
    }

    #[inline]
    pub(crate) fn record(&mut self, route: SchedulerRoute) {
        match route {
            SchedulerRoute::Thread(route) => {
                self.thread_routes += 1;
                self.checksum = mix_route_checksum(
                    self.checksum,
                    1,
                    0,
                    route.process.get(),
                    route.thread.get(),
                    None,
                );
            }
            SchedulerRoute::Process(route) => {
                self.process_routes += 1;
                if route.async_lane.is_some() {
                    self.async_lane_routes += 1;
                }
                self.checksum = mix_route_checksum(
                    self.checksum,
                    2,
                    0,
                    route.process.get(),
                    route.thread.get(),
                    route.async_lane,
                );
            }
            SchedulerRoute::Server(route) => {
                self.server_routes += 1;
                if route.async_lane.is_some() {
                    self.async_lane_routes += 1;
                }
                self.checksum = mix_route_checksum(
                    self.checksum,
                    3,
                    route.server.get(),
                    route.process.get(),
                    route.thread.get(),
                    route.async_lane,
                );
            }
            SchedulerRoute::Accelerator(route) => {
                self.accelerator_routes += 1;
                if route.async_lane.is_some() {
                    self.async_lane_routes += 1;
                }
                match route.kind {
                    AcceleratorKind::Cpu => self.cpu_routes += 1,
                    AcceleratorKind::Gpu => self.gpu_routes += 1,
                    AcceleratorKind::Tpu => self.tpu_routes += 1,
                    AcceleratorKind::Npu => self.npu_routes += 1,
                }
                self.checksum = mix_route_checksum(
                    self.checksum,
                    4,
                    route
                        .kind
                        .checksum_tag()
                        .wrapping_mul(1_009)
                        .wrapping_add(route.accelerator.get()),
                    route.process.get(),
                    route.thread.get(),
                    route.async_lane,
                );
            }
        }
    }
}

#[inline]
fn mix_route_checksum(
    current: usize,
    route_kind: usize,
    server: usize,
    process: usize,
    thread: usize,
    async_lane: Option<AsyncLaneId>,
) -> usize {
    current
        .wrapping_mul(1_000_003)
        .wrapping_add(route_kind.wrapping_mul(97))
        .wrapping_add(server.wrapping_mul(31))
        .wrapping_add(process.wrapping_mul(17))
        .wrapping_add(thread.wrapping_mul(13))
        .wrapping_add(async_lane.map_or(0, |lane| lane.get().wrapping_add(1)))
}
