#[test]
fn transport_archive_channel_keeps_borrowed_views() {
    let source = read_benchmark("../moirai-transport/src/safe_channel.rs");

    for required in [
        "trait ArchiveSerialize",
        "trait ArchiveView",
        "type Archived<'a>",
        "type Archived<'a> = &'a str",
        "impl ArchiveSerialize for str",
        "str::from_utf8(payload)",
        "archive_size_hint",
        "view_start >= buffer_start",
        "view_end <= buffer_end",
        "archive_views_reject_malformed_bytes",
    ] {
        assert!(
            source.contains(required),
            "transport archive channel must retain {required}"
        );
    }

    for prohibited in [
        concat!("Safe", "Serialize"),
        concat!("Safe", "Deserialize"),
        concat!("String::from_utf8(payload.", "to_", "vec())"),
        concat!("String::from_utf8(bytes[4..].", "to_", "vec())"),
    ] {
        assert!(
            !source.contains(prohibited),
            "transport archive channel must not reintroduce {prohibited}"
        );
    }
}

#[test]
fn transport_archive_benchmark_compares_real_borrowed_and_owned_paths() {
    let source = read_benchmark("benches/transport_archive_comparison.rs");

    for required in [
        "borrowed_archive_view",
        "owned_decode_reference",
        "archived_transport_borrowed_view",
        "raw_transport_owned_decode_reference",
        "transport_payload_region_handoff",
        "device_region_owned_handoff",
        "DevicePayloadRegion",
        "payload.handoff::<DevicePayloadRegion>()",
        "ArchivedUniversalSender::<str>",
        "ArchivedUniversalReceiver::<String>",
        "TransportManager::new()",
        "archive_bytes()",
        "String::from_utf8(payload.to_owned())",
        "verify_len",
        "assert_eq!(actual, expected)",
        "without_plots",
    ] {
        assert!(
            source.contains(required),
            "transport archive benchmark must contain {required}"
        );
    }

    let lowered = source.to_lowercase();
    assert!(
        !lowered.contains(&["simu", "lated"].concat())
            && !lowered.contains(&["simu", "lation"].concat()),
        "transport archive benchmark must not contain non-executable benchmark claims"
    );
}

#[test]
fn scheduler_routes_bind_to_archived_transport_without_fake_remote_execution() {
    let manifest = read_benchmark("../moirai-transport/Cargo.toml");
    let lib = read_benchmark("../moirai-transport/src/lib.rs");
    let payload = read_benchmark("../moirai-transport/src/payload.rs");
    let route = read_benchmark("../moirai-transport/src/route.rs");
    let route_tests = read_benchmark("../moirai-transport/src/route/tests.rs");
    let route_all = format!("{payload}\n{route}\n{route_tests}");
    let adr = read_benchmark("../docs/adr.md");
    let checklist = read_benchmark("../docs/adr-008-checklist.md");

    for required in [
        "moirai-executor = { workspace = true, optional = true }",
        "scheduler-routes = [\"moirai-executor\"]",
    ] {
        assert!(
            manifest.contains(required),
            "transport manifest must retain scheduler route feature through {required}"
        );
    }

    assert!(
        lib.contains("#[cfg(feature = \"scheduler-routes\")]\npub mod route;"),
        "transport route consumer must stay feature-gated"
    );

    for required in [
        "pub struct RouteNamespace",
        "pub struct RouteService",
        "pub struct ServerEndpoint",
        "pub struct ProcessEndpoint",
        "pub struct RouteAddressBook",
        "pub struct RoutedArchivedSender<P: RoutePolicy>",
        "pub struct RoutedArchivedReceiver<P: RoutePolicy>",
        "pub struct RoutedRemoteTaskClient<P: RoutePolicy>",
        "pub struct RoutedProcessTaskClient<P: RoutePolicy>",
        "pub enum RoutedProcessTaskError",
        "pub struct RoutedProcessTaskOutput",
        "_policy: PhantomData<P>",
        "pub fn resolve(&self, route: SchedulerRoute) -> Address",
        "pub fn send_route<T>(&self, route: SchedulerRoute, value: &T) -> TransportResult<Address>",
        "pub fn send_selected<C, T>",
        "C: WorkClass",
        "T: ArchiveSerialize + ?Sized",
        "archive_route_payload(route, value)",
        "archive_transport_payload::<ThreadPayloadRegion, T>(value)",
        "payload.handoff::<ProcessPayloadRegion>().into_bytes()",
        "payload.handoff::<ServerPayloadRegion>().into_bytes()",
        "payload.handoff::<DevicePayloadRegion>().into_bytes()",
        "pub fn recv_route<T>(&self, route: SchedulerRoute) -> TransportResult<ArchivedMessage<T>>",
        "pub fn execute_route(",
        "pub fn execute_selected<C>",
        "RemoteTaskClient::new(server, self.reply_to.clone()).execute(task_id, operation)",
        "ProcessSupervisor::new()",
        "supervisor.spawn(endpoint.spec.clone(), self.drop_policy)",
        "SchedulerRoute::Process(route)",
        "RemoteTaskClient::new(endpoint.task_server.clone(), self.reply_to.clone())",
        "process.wait_bounded(self.wait_policy)",
        ".terminate()?",
        "T: ArchiveView",
        "router.select::<C>(priority, sequence)",
        "route-owned archive bytes",
        "accelerator_route_archives_owned_device_payload_bytes",
        "server_route_resolves_to_remote_endpoint_without_sending",
        "routed_remote_task_client_executes_selected_server_route",
        "routed_process_task_client_executes_selected_process_route",
        "process_route_child_serves_one_remote_task",
    ] {
        assert!(
            route_all.contains(required),
            "transport route consumer must retain {required}"
        );
    }

    for prohibited in [
        "dyn RoutePolicy",
        "Box<dyn RoutePolicy",
        "Command::new",
        "TcpStream",
        "tokio::spawn",
        "std::process::Command",
    ] {
        assert!(
            !route.contains(prohibited),
            "transport route consumer must not fake process/server execution through {prohibited}"
        );
    }

    for required in [
        "ADR-008: Scheduler Route Consumption and Transport Ownership Boundary",
        "Route values are metadata until a transport backend consumes them",
        "Mnemosyne allocator handoff is an owned-byte transfer contract",
        "TransportPayload<R>` tags archive bytes with sealed thread, process, server, and device payload regions",
    ] {
        assert!(adr.contains(required), "ADR-008 must retain {required}");
    }

    for required in [
        "Route-to-transport address binding",
        "transport route consumer",
        "OS process executor lifecycle",
        "Mnemosyne allocator ownership handoff",
        "Route-to-remote-task scheduler integration",
        "Route-to-process task execution",
    ] {
        assert!(
            checklist.contains(required),
            "ADR-008 checklist must retain {required}"
        );
    }
}

#[test]
fn remote_transport_uses_real_length_prefixed_tcp_bytes() {
    let lib = read_benchmark("../moirai-transport/src/lib.rs");
    let source = read_benchmark("../moirai-transport/src/network.rs");
    let source_all = format!("{lib}\n{source}");
    let network_start = source
        .find("impl Transport for NetworkTransport {")
        .expect("NetworkTransport impl must exist");
    let network_tail = &source[network_start..];
    let network_end = network_tail
        .find("/// TCP transport for reliable network communication.")
        .expect("TCP transport marker must follow NetworkTransport impl");
    let network_impl = &network_tail[..network_end];
    let adr = read_benchmark("../docs/adr.md");
    let checklist = read_benchmark("../docs/adr-008-checklist.md");

    for required in [
        "mod network;",
        "pub use network::NetworkTransport;",
        "pub(crate) use network::read_network_frame_from_stream;",
    ] {
        assert!(
            lib.contains(required),
            "transport lib must retain network module boundary through {required}"
        );
    }

    for required in [
        "const NETWORK_LENGTH_PREFIX_BYTES: usize = core::mem::size_of::<u64>();",
        "const MAX_NETWORK_MESSAGE_BYTES: u64 = 16 * 1024 * 1024;",
        "pub struct NetworkTransport {}",
        "fn write_network_frame(address: &RemoteAddress, data: &[u8]) -> TransportResult<()>",
        "fn read_network_frame(address: &RemoteAddress) -> TransportResult<Vec<u8>>",
        "pub(crate) fn read_network_frame_from_stream(stream: &mut impl Read) -> TransportResult<Vec<u8>>",
        "fn connect_network_stream(address: &RemoteAddress) -> TransportResult<TcpStream>",
        "const NETWORK_CONNECT_ATTEMPTS: usize = 64;",
        "TcpListener::bind(socket_address(address))",
        ".write_all(&length.to_le_bytes())",
        ".read_exact(&mut length_bytes)",
        "if length > MAX_NETWORK_MESSAGE_BYTES",
        "network_transport_transfers_length_prefixed_remote_bytes",
        "transport_manager_routes_remote_bytes_through_network_transport",
    ] {
        assert!(
            source_all.contains(required),
            "remote transport must retain real TCP byte path through {required}"
        );
    }

    for prohibited in [
        "fn send(&self, _target: &Address, _data: Vec<u8>)",
        "fn send(&self, _target: &Address, data: Vec<u8>)",
        "fn recv(&self, _source: &Address) -> TransportResult<Vec<u8>> {\n        Err(TransportError::Closed)",
    ] {
        assert!(
            !network_impl.contains(prohibited),
            "remote transport must not retain placeholder network path through {prohibited}"
        );
    }

    for required in [
        "`NetworkTransport` sends and receives remote payload bytes through a blocking TCP length-prefixed frame",
        "Remote byte transport is not remote task execution",
    ] {
        assert!(adr.contains(required), "ADR-008 must retain {required}");
    }

    for required in ["Remote byte transport", "Remote task envelopes/results"] {
        assert!(
            checklist.contains(required),
            "ADR-008 checklist must retain {required}"
        );
    }
}

#[test]
fn remote_task_envelopes_execute_value_checked_builtin_operations() {
    let lib = read_benchmark("../moirai-transport/src/lib.rs");
    let payload = read_benchmark("../moirai-transport/src/payload.rs");
    let source = read_benchmark("../moirai-transport/src/remote_task.rs");
    let capability = read_benchmark("../moirai-transport/src/remote_task/capability.rs");
    let server = read_benchmark("../moirai-transport/src/remote_task/server.rs");
    let tests = read_benchmark("../moirai-transport/src/remote_task/tests.rs");
    let source_all = format!("{payload}\n{source}\n{capability}\n{server}\n{tests}");
    let adr = read_benchmark("../docs/adr.md");
    let checklist = read_benchmark("../docs/adr-008-checklist.md");

    assert!(
        lib.contains("pub mod remote_task;"),
        "transport crate must expose remote task envelope module"
    );

    for required in [
        "pub struct RemoteTaskId",
        "mod capability;",
        "pub trait RemoteCapability: sealed::Sealed + Copy + Default + Send + Sync + 'static",
        "const OPERATION_KIND: RemoteTaskOperationKind;",
        "pub enum RemoteTaskOperationKind",
        "pub struct EchoBytesCapability",
        "pub struct SumU64Capability",
        "pub struct RemoteCapabilityToken<C: RemoteCapability>",
        "_capability: PhantomData<C>",
        "pub const fn new() -> Self",
        "pub trait IntoRemoteOperation<C: RemoteCapability>",
        "impl IntoRemoteOperation<EchoBytesCapability> for Vec<u8>",
        "impl IntoRemoteOperation<SumU64Capability> for Vec<u64>",
        "pub fn build_remote_operation<C, P>(",
        "P: IntoRemoteOperation<C>",
        "pub enum RemoteTaskOperation",
        "EchoBytes(Vec<u8>)",
        "SumU64(Vec<u64>)",
        "pub struct RemoteTaskEnvelope",
        "pub enum RemoteTaskOperationView<'a>",
        "EchoBytes(&'a [u8])",
        "pub struct RemoteU64List<'a>",
        "pub fn wrapping_sum(self) -> u64",
        "pub struct RemoteTaskServer",
        "pub struct BoundedRemoteTaskServer",
        "pub struct RemoteTaskQueueCapacity",
        "pub struct RemoteTaskWorkerCount",
        "pub struct RemoteTaskRequestLimit",
        "pub struct RemoteTaskServerStats",
        "pub struct RemoteTaskClient",
        "TransportPayload::<ServerPayloadRegion>::from_bytes(bytes)",
        "archive_transport_payload::<ServerPayloadRegion, _>(&envelope)",
        "archive_transport_payload::<ServerPayloadRegion, _>(&result)",
        "pub fn serve_one(&self) -> TransportResult<RemoteTaskId>",
        "pub fn serve(&self, limit: RemoteTaskRequestLimit) -> TransportResult<RemoteTaskServerStats>",
        "mpsc::sync_channel(self.queue_capacity.get())",
        "TcpListener::bind(remote_socket(&self.bind))",
        "read_network_frame_from_stream(&mut stream)",
        "remote_task_worker(receiver)",
        "pub fn execute(",
        "impl ArchiveSerialize for RemoteTaskEnvelope",
        "impl ArchiveView for RemoteTaskEnvelope",
        "impl ArchiveSerialize for RemoteTaskResult",
        "impl ArchiveView for RemoteTaskResult",
        "fn execute_remote_task(request: &RemoteTaskEnvelopeView<'_>) -> RemoteTaskResult",
        "remote_task_client_server_executes_sum_roundtrip",
        "remote_task_client_server_executes_echo_roundtrip",
        "bounded_remote_task_server_executes_multiple_requests_with_bounded_queue",
        "remote_capability_tokens_are_zero_sized",
        "remote_capabilities_build_only_fixed_format_operations",
        "remote_task_archives_reject_malformed_bytes",
    ] {
        assert!(
            source_all.contains(required),
            "remote task envelope implementation must retain {required}"
        );
    }

    for prohibited in [
        "Box<dyn",
        "dyn RemoteTask",
        "FnOnce",
        "FnMut",
        "dyn RemoteCapability",
        "Command::new",
        "std::process::Command",
        "tokio::spawn",
        "unimplemented!",
        "todo!",
    ] {
        assert!(
            !source_all.contains(prohibited),
            "remote task envelope implementation must not use placeholder or dynamic execution through {prohibited}"
        );
    }

    for required in [
        "Remote task envelopes/results are fixed-format archive contracts",
        "Only explicit built-in operations are executable",
    ] {
        assert!(adr.contains(required), "ADR-008 must retain {required}");
    }

    for required in ["Remote task envelopes/results", "Arbitrary closure remoting"] {
        assert!(
            checklist.contains(required),
            "ADR-008 checklist must retain {required}"
        );
    }
}

#[test]
fn process_supervisor_uses_real_os_process_lifecycle() {
    let lib = read_benchmark("../moirai-transport/src/lib.rs");
    let source = read_benchmark("../moirai-transport/src/process.rs");
    let adr = read_benchmark("../docs/adr.md");
    let checklist = read_benchmark("../docs/adr-008-checklist.md");

    assert!(
        lib.contains("pub mod process;"),
        "transport crate must expose process lifecycle module"
    );

    for required in [
        "process::{Child, Command, ExitStatus}",
        "pub enum ProcessError",
        "SpawnFailed",
        "WaitFailed",
        "TerminateFailed",
        "pub struct ManagedProcessId",
        "pub enum ProcessDropPolicy",
        "TerminateOnDrop",
        "DetachOnDrop",
        "pub struct ProcessWaitPolicy",
        "pub struct ProcessSpec",
        "args: Vec<OsString>",
        "envs: Vec<(OsString, OsString)>",
        "pub fn env(",
        "command.envs(spec.envs)",
        "pub struct ProcessStatus",
        "pub enum ProcessOutcome",
        "Succeeded",
        "Failed",
        "pub struct ProcessSupervisor",
        "pub fn spawn(",
        "Command::new(spec.program)",
        "command.args(spec.args)",
        "command.spawn()",
        "pub struct ManagedProcess",
        "child: Child",
        "pub fn try_wait(&mut self)",
        "pub fn wait(&mut self)",
        "pub fn wait_bounded(",
        "pub fn terminate(&mut self)",
        "self.child.kill()",
        "impl Drop for ManagedProcess",
        "process_supervisor_waits_for_successful_child",
        "process_supervisor_times_out_and_terminates_child",
    ] {
        assert!(
            source.contains(required),
            "process supervisor must retain real lifecycle component {required}"
        );
    }

    for prohibited in [
        "pub success: bool",
        "unimplemented!",
        "todo!",
        "Default::default()",
        "Command::new(\"echo\")",
        "ExitStatus::from_raw",
    ] {
        assert!(
            !source.contains(prohibited),
            "process supervisor must not regress to placeholder lifecycle path through {prohibited}"
        );
    }

    for required in [
        "OS process lifecycle primitives use `ProcessSupervisor`",
        "Process lifecycle is real OS process management",
    ] {
        assert!(adr.contains(required), "ADR-008 must retain {required}");
    }

    assert!(
        checklist.contains("OS process executor lifecycle"),
        "ADR-008 checklist must retain process lifecycle tracking"
    );
}
