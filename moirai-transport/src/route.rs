//! Transport consumers for scheduler route decisions.
//!
//! This module binds concrete scheduler route metadata to transport addresses.
//! It does not spawn processes or open server connections; execution remains
//! owned by the selected transport backend.

use crate::{
    safe_channel::{ArchiveSerialize, ArchiveView, ArchivedMessage},
    Address, RemoteAddress, TransportManager, TransportResult,
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
        self.transport.send(&address, value.archive_bytes()?)?;
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
mod tests {
    use super::{RouteAddressBook, RouteNamespace, RouteService, RoutedArchivedReceiver};
    use crate::route::{RoutedArchivedSender, ServerEndpoint};
    use crate::Address;
    use moirai_core::Priority;
    use moirai_executor::schedule::{
        AsyncLanesPerProcess, AsyncTask, HybridRoutePolicy, HybridRouter, ProcessCount,
        RouteTopology, SchedulerRoute, ServerCount, ServerId, ServerRoutePolicy, ThreadRoutePolicy,
        WorkerCount,
    };
    use std::sync::Arc;

    fn topology(servers: usize) -> RouteTopology {
        RouteTopology::new(
            WorkerCount::new(4),
            ProcessCount::new(3),
            AsyncLanesPerProcess::new(2),
            ServerCount::new(servers),
        )
    }

    fn address_book() -> RouteAddressBook {
        RouteAddressBook::new(
            RouteNamespace::new("scheduler-route"),
            vec![ServerEndpoint::new(
                ServerId::new(0),
                "127.0.0.1",
                9700,
                RouteService::new("moirai-route"),
            )],
        )
    }

    #[test]
    fn routed_archived_sender_roundtrips_local_thread_route() {
        let transport = Arc::new(crate::TransportManager::new());
        let router = HybridRouter::<ThreadRoutePolicy>::new(topology(0));
        let sender =
            RoutedArchivedSender::<ThreadRoutePolicy>::new(Arc::clone(&transport), address_book());
        let receiver = RoutedArchivedReceiver::<ThreadRoutePolicy>::new(transport, address_book());
        let value = String::from("route-owned archive bytes");

        let route = sender
            .send_selected::<AsyncTask, str>(&router, Priority::Normal, 7, value.as_str())
            .unwrap();
        let message = receiver.recv_route::<String>(route).unwrap();

        assert_eq!(message.get().unwrap(), value.as_str());
    }

    #[test]
    fn async_process_route_resolves_to_async_lane_address() {
        let router = HybridRouter::<HybridRoutePolicy>::new(topology(0));
        let route = router.select::<AsyncTask>(Priority::High, 5);
        let address = address_book().resolve(route);

        match address {
            Address::Local(address) => {
                assert!(address.contains("/process/"));
                assert!(address.contains("/thread/"));
                assert!(address.contains("/async-lane/"));
            }
            Address::Remote(_) => panic!("process route must resolve locally without servers"),
        }
    }

    #[test]
    fn server_route_resolves_to_remote_endpoint_without_sending() {
        let router = HybridRouter::<ServerRoutePolicy>::new(topology(1));
        let route = (0..64)
            .map(|sequence| router.select::<AsyncTask>(Priority::Critical, sequence))
            .find(|route| matches!(route, SchedulerRoute::Server(_)))
            .expect("test topology must produce a server route");
        let address = address_book().resolve(route);

        match address {
            Address::Remote(remote) => {
                assert_eq!(remote.host, "127.0.0.1");
                assert_eq!(remote.port, 9700);
                assert_eq!(remote.service, "moirai-route");
            }
            Address::Local(_) => panic!("known server route must resolve remotely"),
        }
    }
}
