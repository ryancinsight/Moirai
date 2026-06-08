//! Transport consumers for scheduler route decisions.
//!
//! This module binds concrete scheduler route metadata to transport addresses.
//! It does not spawn processes or open server connections; execution remains
//! owned by the selected transport backend.

use crate::{
    payload::{
        archive_transport_payload, ProcessPayloadRegion, ServerPayloadRegion, ThreadPayloadRegion,
    },
    process::{
        ManagedProcessId, ProcessDropPolicy, ProcessError, ProcessSpec, ProcessStatus,
        ProcessSupervisor, ProcessWaitPolicy,
    },
    remote_task::{RemoteTaskClient, RemoteTaskId, RemoteTaskOperation, RemoteTaskResult},
    safe_channel::{ArchiveSerialize, ArchiveView, ArchivedMessage},
    Address, RemoteAddress, TransportError, TransportManager, TransportResult,
};
use moirai_core::Priority;
use moirai_executor::schedule::{
    ProcessId, RoutePolicy, SchedulerRoute, ServerId, ThreadId, WorkClass,
};
use std::{marker::PhantomData, sync::Arc};

/// Local address namespace for routed scheduler messages.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RouteNamespace(String);

impl RouteNamespace {
    /// Construct a route namespace.
    pub fn new(namespace: impl Into<String>) -> Self {
        Self(namespace.into())
    }

    /// Borrow the namespace text.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Service label for remote server routes.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RouteService(String);

impl RouteService {
    /// Construct a route service label.
    pub fn new(service: impl Into<String>) -> Self {
        Self(service.into())
    }

    /// Borrow the service text.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Static remote endpoint catalog for server routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerEndpoint {
    /// Server identifier from the scheduler route.
    pub server: ServerId,
    /// Remote host name or IP address.
    pub host: String,
    /// Remote port.
    pub port: u16,
    /// Remote service label.
    pub service: RouteService,
}

impl ServerEndpoint {
    /// Construct a server endpoint.
    pub fn new(
        server: ServerId,
        host: impl Into<String>,
        port: u16,
        service: RouteService,
    ) -> Self {
        Self {
            server,
            host: host.into(),
            port,
            service,
        }
    }
}

/// Supervised child process endpoint for process routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessEndpoint {
    /// Process identifier from the scheduler route.
    pub process: ProcessId,
    /// Child process command specification.
    pub spec: ProcessSpec,
    /// Remote task server address served by the child process.
    pub task_server: RemoteAddress,
}

impl ProcessEndpoint {
    /// Construct a process endpoint.
    pub fn new(process: ProcessId, spec: ProcessSpec, task_server: RemoteAddress) -> Self {
        Self {
            process,
            spec,
            task_server,
        }
    }
}

/// Route-to-address resolver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteAddressBook {
    namespace: RouteNamespace,
    servers: Vec<ServerEndpoint>,
}

impl RouteAddressBook {
    /// Construct a route address book.
    pub fn new(namespace: RouteNamespace, servers: Vec<ServerEndpoint>) -> Self {
        Self { namespace, servers }
    }

    /// Resolve a scheduler route into a transport address.
    pub fn resolve(&self, route: SchedulerRoute) -> Address {
        match route {
            SchedulerRoute::Thread(route) => Address::Local(local_address(
                self.namespace.as_str(),
                route.process,
                route.thread,
                None,
            )),
            SchedulerRoute::Process(route) => Address::Local(local_address(
                self.namespace.as_str(),
                route.process,
                route.thread,
                route.async_lane.map(|lane| lane.get()),
            )),
            SchedulerRoute::Server(route) => self
                .servers
                .iter()
                .find(|endpoint| endpoint.server == route.server)
                .map(|endpoint| {
                    Address::Remote(RemoteAddress {
                        host: endpoint.host.clone(),
                        port: endpoint.port,
                        service: endpoint.service.as_str().to_string(),
                    })
                })
                .unwrap_or_else(|| {
                    Address::Local(local_address(
                        self.namespace.as_str(),
                        route.process,
                        route.thread,
                        route.async_lane.map(|lane| lane.get()),
                    ))
                }),
        }
    }
}

/// Failure modes for routed process task execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedProcessTaskError {
    /// Selected route was not a process route.
    NonProcessRoute,
    /// No endpoint is registered for the selected process route.
    MissingProcessEndpoint,
    /// Child process lifecycle failed.
    Process(ProcessError),
    /// Remote task transport failed.
    Transport(TransportError),
}

impl From<ProcessError> for RoutedProcessTaskError {
    fn from(error: ProcessError) -> Self {
        Self::Process(error)
    }
}

impl From<TransportError> for RoutedProcessTaskError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}

/// Result type for routed process task execution.
pub type RoutedProcessTaskResult<T> = Result<T, RoutedProcessTaskError>;

/// Value result from a supervised process-routed remote task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedProcessTaskOutput {
    /// Child process id.
    pub process_id: ManagedProcessId,
    /// Remote task result produced by the child process server.
    pub result: RemoteTaskResult,
    /// Child process completion status.
    pub status: ProcessStatus,
}

/// Archived transport sender driven by scheduler route decisions.
pub struct RoutedArchivedSender<P: RoutePolicy> {
    transport: Arc<TransportManager>,
    address_book: RouteAddressBook,
    _policy: PhantomData<P>,
}

impl<P: RoutePolicy> RoutedArchivedSender<P> {
    /// Construct a routed archived sender.
    pub fn new(transport: Arc<TransportManager>, address_book: RouteAddressBook) -> Self {
        Self {
            transport,
            address_book,
            _policy: PhantomData,
        }
    }

    /// Archive and send a value to the address selected by `route`.
    pub fn send_route<T>(&self, route: SchedulerRoute, value: &T) -> TransportResult<Address>
    where
        T: ArchiveSerialize + ?Sized,
    {
        let address = self.address_book.resolve(route);
        self.transport
            .send(&address, archive_route_payload(route, value)?)?;
        Ok(address)
    }

    /// Archive and send a value after selecting a route from `router`.
    pub fn send_selected<C, T>(
        &self,
        router: &moirai_executor::schedule::HybridRouter<P>,
        priority: Priority,
        sequence: usize,
        value: &T,
    ) -> TransportResult<SchedulerRoute>
    where
        C: WorkClass,
        T: ArchiveSerialize + ?Sized,
    {
        let route = router.select::<C>(priority, sequence);
        self.send_route(route, value)?;
        Ok(route)
    }
}

fn archive_route_payload<T>(route: SchedulerRoute, value: &T) -> TransportResult<Vec<u8>>
where
    T: ArchiveSerialize + ?Sized,
{
    let payload = archive_transport_payload::<ThreadPayloadRegion, T>(value)?;
    Ok(match route {
        SchedulerRoute::Thread(_) => payload.into_bytes(),
        SchedulerRoute::Process(_) => payload.handoff::<ProcessPayloadRegion>().into_bytes(),
        SchedulerRoute::Server(_) => payload.handoff::<ServerPayloadRegion>().into_bytes(),
    })
}

/// Archived transport receiver driven by scheduler route decisions.
pub struct RoutedArchivedReceiver<P: RoutePolicy> {
    transport: Arc<TransportManager>,
    address_book: RouteAddressBook,
    _policy: PhantomData<P>,
}

impl<P: RoutePolicy> RoutedArchivedReceiver<P> {
    /// Construct a routed archived receiver.
    pub fn new(transport: Arc<TransportManager>, address_book: RouteAddressBook) -> Self {
        Self {
            transport,
            address_book,
            _policy: PhantomData,
        }
    }

    /// Receive transport bytes from the address selected by `route`.
    pub fn recv_route<T>(&self, route: SchedulerRoute) -> TransportResult<ArchivedMessage<T>>
    where
        T: ArchiveView,
    {
        self.transport
            .recv(&self.address_book.resolve(route))
            .map(ArchivedMessage::from_bytes)
    }
}

/// Remote task client driven by scheduler route decisions.
pub struct RoutedRemoteTaskClient<P: RoutePolicy> {
    address_book: RouteAddressBook,
    reply_to: RemoteAddress,
    _policy: PhantomData<P>,
}

impl<P: RoutePolicy> RoutedRemoteTaskClient<P> {
    /// Construct a routed remote task client.
    pub fn new(address_book: RouteAddressBook, reply_to: RemoteAddress) -> Self {
        Self {
            address_book,
            reply_to,
            _policy: PhantomData,
        }
    }

    /// Execute a remote task against an already selected scheduler route.
    pub fn execute_route(
        &self,
        route: SchedulerRoute,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> TransportResult<RemoteTaskResult> {
        let Address::Remote(server) = self.address_book.resolve(route) else {
            return Err(TransportError::Closed);
        };

        RemoteTaskClient::new(server, self.reply_to.clone()).execute(task_id, operation)
    }

    /// Select a scheduler route and execute a fixed-format remote task there.
    pub fn execute_selected<C>(
        &self,
        router: &moirai_executor::schedule::HybridRouter<P>,
        priority: Priority,
        sequence: usize,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> TransportResult<(SchedulerRoute, RemoteTaskResult)>
    where
        C: WorkClass,
    {
        let route = router.select::<C>(priority, sequence);
        let result = self.execute_route(route, task_id, operation)?;
        Ok((route, result))
    }
}

/// Remote task client that launches a supervised child process for process routes.
pub struct RoutedProcessTaskClient<P: RoutePolicy> {
    endpoints: Vec<ProcessEndpoint>,
    reply_to: RemoteAddress,
    drop_policy: ProcessDropPolicy,
    wait_policy: ProcessWaitPolicy,
    _policy: PhantomData<P>,
}

impl<P: RoutePolicy> RoutedProcessTaskClient<P> {
    /// Construct a routed process task client.
    pub fn new(
        endpoints: Vec<ProcessEndpoint>,
        reply_to: RemoteAddress,
        drop_policy: ProcessDropPolicy,
        wait_policy: ProcessWaitPolicy,
    ) -> Self {
        Self {
            endpoints,
            reply_to,
            drop_policy,
            wait_policy,
            _policy: PhantomData,
        }
    }

    /// Execute a fixed-format remote task through a selected process route.
    pub fn execute_route(
        &self,
        route: SchedulerRoute,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> RoutedProcessTaskResult<RoutedProcessTaskOutput> {
        let SchedulerRoute::Process(route) = route else {
            return Err(RoutedProcessTaskError::NonProcessRoute);
        };

        let endpoint = self
            .endpoints
            .iter()
            .find(|endpoint| endpoint.process == route.process)
            .ok_or(RoutedProcessTaskError::MissingProcessEndpoint)?;
        let supervisor = ProcessSupervisor::new();
        let mut process = supervisor.spawn(endpoint.spec.clone(), self.drop_policy)?;
        let process_id = process.id();
        let result = RemoteTaskClient::new(endpoint.task_server.clone(), self.reply_to.clone())
            .execute(task_id, operation)?;
        let status = match process.wait_bounded(self.wait_policy)? {
            Some(status) => status,
            None => process
                .terminate()?
                .ok_or(RoutedProcessTaskError::Process(ProcessError::WaitFailed))?,
        };

        Ok(RoutedProcessTaskOutput {
            process_id,
            result,
            status,
        })
    }

    /// Select a process route and execute a fixed-format remote task through it.
    pub fn execute_selected<C>(
        &self,
        router: &moirai_executor::schedule::HybridRouter<P>,
        priority: Priority,
        sequence: usize,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> RoutedProcessTaskResult<(SchedulerRoute, RoutedProcessTaskOutput)>
    where
        C: WorkClass,
    {
        let route = router.select::<C>(priority, sequence);
        let output = self.execute_route(route, task_id, operation)?;
        Ok((route, output))
    }
}

fn local_address(
    namespace: &str,
    process: ProcessId,
    thread: ThreadId,
    async_lane: Option<usize>,
) -> String {
    match async_lane {
        Some(lane) => format!(
            "{namespace}/process/{}/thread/{}/async-lane/{lane}",
            process.get(),
            thread.get()
        ),
        None => format!(
            "{namespace}/process/{}/thread/{}",
            process.get(),
            thread.get()
        ),
    }
}

#[cfg(test)]
mod tests;
