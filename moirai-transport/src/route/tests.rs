use super::{
    ProcessEndpoint, RouteAddressBook, RouteNamespace, RouteService, RoutedArchivedReceiver,
    RoutedProcessTaskClient, RoutedRemoteTaskClient,
};
use crate::route::{RoutedArchivedSender, ServerEndpoint};
use crate::{
    process::{ProcessDropPolicy, ProcessOutcome, ProcessSpec, ProcessWaitPolicy},
    remote_task::{RemoteTaskId, RemoteTaskOperation, RemoteTaskOutput, RemoteTaskServer},
    Address, RemoteAddress,
};
use moirai_core::Priority;
use moirai_executor::schedule::{
    AsyncLanesPerProcess, AsyncTask, HybridRoutePolicy, HybridRouter, ProcessCount, RouteTopology,
    SchedulerRoute, ServerCount, ServerId, ServerRoutePolicy, ThreadRoutePolicy, WorkerCount,
};
use std::{sync::Arc, time::Duration};

fn topology(servers: usize) -> RouteTopology {
    RouteTopology::new(
        WorkerCount::new(4),
        ProcessCount::new(3),
        AsyncLanesPerProcess::new(2),
        ServerCount::new(servers),
    )
}

fn address_book() -> RouteAddressBook {
    address_book_for_port(9700)
}

fn address_book_for_port(port: u16) -> RouteAddressBook {
    RouteAddressBook::new(
        RouteNamespace::new("scheduler-route"),
        vec![ServerEndpoint::new(
            ServerId::new(0),
            "127.0.0.1",
            port,
            RouteService::new("moirai-route"),
        )],
    )
}

fn loopback_remote_address() -> RemoteAddress {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    RemoteAddress {
        host: "127.0.0.1".to_string(),
        port,
        service: "moirai-route".to_string(),
    }
}

fn process_route_sequence(
    router: &HybridRouter<HybridRoutePolicy>,
) -> (usize, moirai_executor::schedule::ProcessId) {
    (0..64)
        .find_map(
            |sequence| match router.select::<AsyncTask>(Priority::High, sequence) {
                SchedulerRoute::Process(route) => Some((sequence, route.process)),
                SchedulerRoute::Thread(_) | SchedulerRoute::Server(_) => None,
            },
        )
        .expect("test topology must produce a process route")
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

#[test]
fn routed_remote_task_client_executes_selected_server_route() {
    let server_address = loopback_remote_address();
    let reply_address = loopback_remote_address();
    let address_book = address_book_for_port(server_address.port);
    let router = HybridRouter::<ServerRoutePolicy>::new(topology(1));
    let sequence = (0..64)
        .find(|sequence| {
            matches!(
                router.select::<AsyncTask>(Priority::Critical, *sequence),
                SchedulerRoute::Server(_)
            )
        })
        .expect("test topology must produce a server route");
    let server = RemoteTaskServer::new(server_address);
    let server_thread = std::thread::spawn(move || server.serve_one().unwrap());
    let client = RoutedRemoteTaskClient::<ServerRoutePolicy>::new(address_book, reply_address);
    let task_id = RemoteTaskId::new(41);

    let (route, result) = client
        .execute_selected::<AsyncTask>(
            &router,
            Priority::Critical,
            sequence,
            task_id,
            RemoteTaskOperation::SumU64(vec![21, 34, 55]),
        )
        .unwrap();

    assert!(matches!(route, SchedulerRoute::Server(_)));
    assert_eq!(server_thread.join().unwrap(), task_id);
    assert_eq!(result.output, RemoteTaskOutput::U64(110));
}

#[test]
fn routed_process_task_client_executes_selected_process_route() {
    let server_address = loopback_remote_address();
    let reply_address = loopback_remote_address();
    let router = HybridRouter::<HybridRoutePolicy>::new(topology(0));
    let (sequence, process) = process_route_sequence(&router);
    let task_id = RemoteTaskId::new(53);
    let spec = ProcessSpec::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "route::tests::process_route_child_serves_one_remote_task",
            "--nocapture",
        ])
        .env("MOIRAI_PROCESS_TASK_PORT", server_address.port.to_string())
        .env("MOIRAI_PROCESS_TASK_ID", task_id.get().to_string());
    let endpoint = ProcessEndpoint::new(process, spec, server_address);
    let client = RoutedProcessTaskClient::<HybridRoutePolicy>::new(
        vec![endpoint],
        reply_address,
        ProcessDropPolicy::TerminateOnDrop,
        ProcessWaitPolicy::new(1_000, Duration::from_millis(1)),
    );

    let (route, output) = client
        .execute_selected::<AsyncTask>(
            &router,
            Priority::High,
            sequence,
            task_id,
            RemoteTaskOperation::SumU64(vec![89, 144, 233]),
        )
        .unwrap();

    assert!(matches!(route, SchedulerRoute::Process(_)));
    assert_ne!(output.process_id.get(), 0);
    assert_eq!(output.result.task_id, task_id);
    assert_eq!(output.result.output, RemoteTaskOutput::U64(466));
    assert_eq!(output.status.outcome, ProcessOutcome::Succeeded);
    assert_eq!(output.status.code, Some(0));
}

#[test]
#[ignore]
fn process_route_child_serves_one_remote_task() {
    let port = std::env::var("MOIRAI_PROCESS_TASK_PORT")
        .expect("process task port must be provided")
        .parse::<u16>()
        .expect("process task port must be a u16");
    let expected_task = std::env::var("MOIRAI_PROCESS_TASK_ID")
        .expect("process task id must be provided")
        .parse::<u64>()
        .expect("process task id must be a u64");
    let server = RemoteTaskServer::new(RemoteAddress {
        host: "127.0.0.1".to_string(),
        port,
        service: "moirai-process".to_string(),
    });

    assert_eq!(
        server.serve_one().unwrap(),
        RemoteTaskId::new(expected_task)
    );
}
