//! Zero-sized route policy markers.

mod route_policy {
    pub trait Sealed {}
}

/// Compile-time scheduler route policy.
///
/// Implementors are zero-sized marker types. The router uses the associated
/// constants as structural branch selectors, so inactive routing families are
/// removed by monomorphization and dead-code elimination.
pub trait RoutePolicy: route_policy::Sealed + Copy + Default + Send + Sync + 'static {
    /// Whether route selection may produce process routes.
    const ENABLE_PROCESS_ROUTES: bool;

    /// Whether route selection may produce server routes.
    const ENABLE_SERVER_ROUTES: bool;

    /// Whether route selection may produce accelerator metadata routes.
    const ENABLE_ACCELERATOR_ROUTES: bool;

    /// Periodic process route cadence for non-async work classes.
    const PROCESS_PERIOD: usize;

    /// Periodic server route cadence.
    const SERVER_PERIOD: usize;

    /// Periodic accelerator route cadence.
    const ACCELERATOR_PERIOD: usize;
}

/// Route policy that keeps work local to scheduler threads.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadRoutePolicy;

/// Route policy that mixes local threads, process routes, and server routes.
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridRoutePolicy;

/// Route policy that gives server routing a higher cadence.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServerRoutePolicy;

/// Route policy that emits accelerator metadata routes.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcceleratorRoutePolicy;

impl route_policy::Sealed for ThreadRoutePolicy {}
impl route_policy::Sealed for HybridRoutePolicy {}
impl route_policy::Sealed for ServerRoutePolicy {}
impl route_policy::Sealed for AcceleratorRoutePolicy {}

impl RoutePolicy for ThreadRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = false;
    const ENABLE_SERVER_ROUTES: bool = false;
    const ENABLE_ACCELERATOR_ROUTES: bool = false;
    const PROCESS_PERIOD: usize = 1;
    const SERVER_PERIOD: usize = 1;
    const ACCELERATOR_PERIOD: usize = 1;
}

impl RoutePolicy for HybridRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = true;
    const ENABLE_SERVER_ROUTES: bool = true;
    const ENABLE_ACCELERATOR_ROUTES: bool = false;
    const PROCESS_PERIOD: usize = 5;
    const SERVER_PERIOD: usize = 17;
    const ACCELERATOR_PERIOD: usize = 1;
}

impl RoutePolicy for ServerRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = true;
    const ENABLE_SERVER_ROUTES: bool = true;
    const ENABLE_ACCELERATOR_ROUTES: bool = false;
    const PROCESS_PERIOD: usize = 3;
    const SERVER_PERIOD: usize = 11;
    const ACCELERATOR_PERIOD: usize = 1;
}

impl RoutePolicy for AcceleratorRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = true;
    const ENABLE_SERVER_ROUTES: bool = true;
    const ENABLE_ACCELERATOR_ROUTES: bool = true;
    const PROCESS_PERIOD: usize = 5;
    const SERVER_PERIOD: usize = 17;
    const ACCELERATOR_PERIOD: usize = 7;
}
