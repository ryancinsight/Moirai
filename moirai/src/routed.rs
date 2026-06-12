//! Public fixed-capability routed execution facade.

use crate::Moirai;
use moirai_core::Priority;
use moirai_executor::schedule::{HybridRouter, RoutePolicy, SchedulerRoute, WorkClass};
use moirai_transport::{
    process::{ProcessDropPolicy, ProcessWaitPolicy},
    remote_task::{
        build_remote_operation, IntoRemoteOperation, RemoteCapability, RemoteCapabilityToken,
        RemoteTaskId, RemoteTaskResult,
    },
    route::{
        ProcessEndpoint, RouteAddressBook, RoutedProcessTaskClient, RoutedProcessTaskError,
        RoutedProcessTaskOutput, RoutedRemoteTaskClient,
    },
    RemoteAddress, TransportResult,
};

/// Fixed-format routed task admitted by a sealed remote capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRemoteTask<C: RemoteCapability, Payload> {
    task_id: RemoteTaskId,
    token: RemoteCapabilityToken<C>,
    payload: Payload,
}

impl<C: RemoteCapability, Payload> FixedRemoteTask<C, Payload> {
    /// Construct a fixed-format remote task request.
    pub const fn new(
        task_id: RemoteTaskId,
        token: RemoteCapabilityToken<C>,
        payload: Payload,
    ) -> Self {
        Self {
            task_id,
            token,
            payload,
        }
    }
}

/// Server route execution target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedServerTarget {
    address_book: RouteAddressBook,
    reply_to: RemoteAddress,
}

impl RoutedServerTarget {
    /// Construct a server route execution target.
    pub fn new(address_book: RouteAddressBook, reply_to: RemoteAddress) -> Self {
        Self {
            address_book,
            reply_to,
        }
    }
}

/// Process route execution target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedProcessTarget {
    endpoints: Vec<ProcessEndpoint>,
    reply_to: RemoteAddress,
    drop_policy: ProcessDropPolicy,
    wait_policy: ProcessWaitPolicy,
}

impl RoutedProcessTarget {
    /// Construct a process route execution target.
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
        }
    }
}

impl Moirai {
    /// Execute a fixed-format task through a selected server route.
    ///
    /// The public facade accepts only sealed `RemoteCapabilityToken<C>` values
    /// and payloads that can build the matching fixed-format operation. It does
    /// not accept arbitrary Rust closures, dynamic remote tasks, or node
    /// discovery placeholders.
    ///
    /// # Errors
    ///
    /// Returns a transport error if route resolution, request send, reply
    /// receive, or result validation fails.
    pub fn execute_routed_server_task<W, P, C, Payload>(
        &self,
        router: &HybridRouter<P>,
        priority: Priority,
        sequence: usize,
        target: RoutedServerTarget,
        task: FixedRemoteTask<C, Payload>,
    ) -> TransportResult<(SchedulerRoute, RemoteTaskResult)>
    where
        W: WorkClass,
        P: RoutePolicy,
        C: RemoteCapability,
        Payload: IntoRemoteOperation<C>,
    {
        let operation = build_remote_operation(task.payload, task.token);
        let client = RoutedRemoteTaskClient::<P>::new(target.address_book, target.reply_to);

        client.execute_selected::<W>(router, priority, sequence, task.task_id, operation)
    }

    /// Execute a fixed-format task through a selected process route.
    ///
    /// The facade delegates process lifecycle and request/response transport to
    /// `moirai-transport`; this method only binds the public runtime facade to
    /// the sealed fixed-format capability surface.
    ///
    /// # Errors
    ///
    /// Returns a routed process task error if route selection does not produce a
    /// process route, if no endpoint is registered, if process lifecycle fails,
    /// or if request/response transport fails.
    pub fn execute_routed_process_task<W, P, C, Payload>(
        &self,
        router: &HybridRouter<P>,
        priority: Priority,
        sequence: usize,
        target: RoutedProcessTarget,
        task: FixedRemoteTask<C, Payload>,
    ) -> Result<(SchedulerRoute, RoutedProcessTaskOutput), RoutedProcessTaskError>
    where
        W: WorkClass,
        P: RoutePolicy,
        C: RemoteCapability,
        Payload: IntoRemoteOperation<C>,
    {
        let operation = build_remote_operation(task.payload, task.token);
        let client = RoutedProcessTaskClient::<P>::new(
            target.endpoints,
            target.reply_to,
            target.drop_policy,
            target.wait_policy,
        );

        client.execute_selected::<W>(router, priority, sequence, task.task_id, operation)
    }
}

#[cfg(test)]
mod tests {
    use super::{FixedRemoteTask, RoutedProcessTarget, RoutedServerTarget};
    use crate::Moirai;
    use moirai_core::Priority;
    use moirai_executor::schedule::{
        AsyncLanesPerProcess, AsyncTask, HybridRoutePolicy, HybridRouter, ProcessCount,
        RouteTopology, SchedulerRoute, ServerCount, ServerId, ServerRoutePolicy, WorkerCount,
    };
    use moirai_transport::{
        process::{ProcessDropPolicy, ProcessOutcome, ProcessSpec, ProcessWaitPolicy},
        remote_task::{
            RemoteCapabilityToken, RemoteTaskId, RemoteTaskOutput, RemoteTaskServer,
            SumU64Capability,
        },
        route::{ProcessEndpoint, RouteAddressBook, RouteNamespace, RouteService, ServerEndpoint},
        RemoteAddress,
    };
    use std::time::Duration;

    const CHILD_PORT: &str = "MOIRAI_PUBLIC_ROUTED_PROCESS_PORT";
    const CHILD_TASK_ID: &str = "MOIRAI_PUBLIC_ROUTED_PROCESS_TASK_ID";

    fn topology(servers: usize) -> RouteTopology {
        RouteTopology::new(
            WorkerCount::new(4),
            ProcessCount::new(3),
            AsyncLanesPerProcess::new(2),
            ServerCount::new(servers),
        )
    }

    fn loopback_remote_address(service: &'static str) -> RemoteAddress {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        RemoteAddress {
            host: "127.0.0.1".to_string(),
            port,
            service: service.to_string(),
        }
    }

    fn address_book_for(server_address: &RemoteAddress) -> RouteAddressBook {
        RouteAddressBook::new(
            RouteNamespace::new("public-routed-facade"),
            vec![ServerEndpoint::new(
                ServerId::new(0),
                server_address.host.clone(),
                server_address.port,
                RouteService::new(server_address.service.clone()),
            )],
        )
    }

    fn server_route_sequence(router: &HybridRouter<ServerRoutePolicy>) -> usize {
        (0..64)
            .find(|sequence| {
                matches!(
                    router.select::<AsyncTask>(Priority::Critical, *sequence),
                    SchedulerRoute::Server(_)
                )
            })
            .expect("test topology must produce a server route")
    }

    fn process_route_sequence(
        router: &HybridRouter<HybridRoutePolicy>,
    ) -> (usize, moirai_executor::schedule::ProcessId) {
        (0..64)
            .find_map(
                |sequence| match router.select::<AsyncTask>(Priority::High, sequence) {
                    SchedulerRoute::Process(route) => Some((sequence, route.process)),
                    SchedulerRoute::Thread(_)
                    | SchedulerRoute::Server(_)
                    | SchedulerRoute::Accelerator(_) => None,
                },
            )
            .expect("test topology must produce a process route")
    }

    #[test]
    fn public_facade_executes_fixed_capability_server_route() {
        let runtime = Moirai::new().unwrap();
        let server_address = loopback_remote_address("public-routed-server");
        let reply_address = loopback_remote_address("public-routed-reply");
        let router = HybridRouter::<ServerRoutePolicy>::new(topology(1));
        let sequence = server_route_sequence(&router);
        let task_id = RemoteTaskId::new(70);
        let server = RemoteTaskServer::new(server_address.clone());
        let server_thread = std::thread::spawn(move || server.serve_one().unwrap());
        let target = RoutedServerTarget::new(address_book_for(&server_address), reply_address);
        let task = FixedRemoteTask::new(
            task_id,
            RemoteCapabilityToken::<SumU64Capability>::new(),
            vec![1, 2, 3, 4],
        );

        let (route, result) = runtime
            .execute_routed_server_task::<AsyncTask, ServerRoutePolicy, _, _>(
                &router,
                Priority::Critical,
                sequence,
                target,
                task,
            )
            .unwrap();

        assert!(matches!(route, SchedulerRoute::Server(_)));
        assert_eq!(server_thread.join().unwrap(), task_id);
        assert_eq!(result.output, RemoteTaskOutput::U64(10));
        runtime.shutdown();
    }

    #[test]
    fn public_facade_executes_fixed_capability_process_route() {
        let runtime = Moirai::new().unwrap();
        let server_address = loopback_remote_address("public-routed-process");
        let reply_address = loopback_remote_address("public-routed-process-reply");
        let router = HybridRouter::<HybridRoutePolicy>::new(topology(0));
        let (sequence, process) = process_route_sequence(&router);
        let task_id = RemoteTaskId::new(71);
        let spec = ProcessSpec::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "routed::tests::public_routed_process_child_serves_one_task",
                "--nocapture",
            ])
            .env(CHILD_PORT, server_address.port.to_string())
            .env(CHILD_TASK_ID, task_id.get().to_string());
        let target = RoutedProcessTarget::new(
            vec![ProcessEndpoint::new(process, spec, server_address)],
            reply_address,
            ProcessDropPolicy::TerminateOnDrop,
            ProcessWaitPolicy::new(1_000, Duration::from_millis(1)),
        );
        let task = FixedRemoteTask::new(
            task_id,
            RemoteCapabilityToken::<SumU64Capability>::new(),
            vec![8, 13, 21],
        );

        let (route, output) = runtime
            .execute_routed_process_task::<AsyncTask, HybridRoutePolicy, _, _>(
                &router,
                Priority::High,
                sequence,
                target,
                task,
            )
            .unwrap();

        assert!(matches!(route, SchedulerRoute::Process(_)));
        assert_eq!(output.result.task_id, task_id);
        assert_eq!(output.result.output, RemoteTaskOutput::U64(42));
        assert_eq!(output.status.outcome, ProcessOutcome::Succeeded);
        runtime.shutdown();
    }

    #[test]
    #[ignore]
    fn public_routed_process_child_serves_one_task() {
        let port = std::env::var(CHILD_PORT)
            .expect("process task port must be provided")
            .parse::<u16>()
            .expect("process task port must be a u16");
        let expected_task = std::env::var(CHILD_TASK_ID)
            .expect("process task id must be provided")
            .parse::<u64>()
            .expect("process task id must be a u64");
        let server = RemoteTaskServer::new(RemoteAddress {
            host: "127.0.0.1".to_string(),
            port,
            service: "public-routed-process".to_string(),
        });

        assert_eq!(
            server.serve_one().unwrap(),
            RemoteTaskId::new(expected_task)
        );
    }
}
