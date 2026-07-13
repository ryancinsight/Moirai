#[test]
fn transport_payload_regions_define_mnemosyne_handoff_boundary() {
    let lib = read_benchmark("../moirai-transport/src/lib.rs");
    let payload = read_benchmark("../moirai-transport/src/payload.rs");
    let route = read_benchmark("../moirai-transport/src/route.rs");
    let remote_task = read_benchmark("../moirai-transport/src/remote_task.rs");
    let server = read_benchmark("../moirai-transport/src/remote_task/server.rs");
    let moirai = read_benchmark("../moirai/src/lib.rs");
    let moirai_manifest = read_benchmark("../moirai/Cargo.toml");
    let source_all = format!("{payload}\n{route}\n{remote_task}\n{server}");

    assert!(
        lib.contains("pub mod payload;"),
        "transport crate must expose payload ownership regions"
    );

    for required in [
        "pub enum PayloadBoundary",
        "Thread",
        "Process",
        "Server",
        "Device",
        "pub trait PayloadRegion: sealed::Sealed + Copy + Default + Send + Sync + 'static",
        "const POINTER_TRANSFER_ALLOWED: bool",
        "pub struct ThreadPayloadRegion",
        "pub struct ProcessPayloadRegion",
        "pub struct ServerPayloadRegion",
        "pub struct DevicePayloadRegion",
        "pub struct TransportPayload<R: PayloadRegion>",
        "_region: PhantomData<R>",
        "pub fn handoff<Target: PayloadRegion>(self) -> TransportPayload<Target>",
        "pub fn archive_transport_payload<R, T>",
        "T: ArchiveSerialize + ?Sized",
        "payload_region_markers_are_zero_sized",
        "payload_regions_encode_pointer_transfer_contract",
        "payload_handoff_moves_same_owned_buffer_between_regions",
        "archive_transport_payload_preserves_value_bytes",
    ] {
        assert!(
            payload.contains(required),
            "payload region boundary must retain {required}"
        );
    }

    for required in [
        "archive_route_payload(route, value)",
        "payload.handoff::<ProcessPayloadRegion>().into_bytes()",
        "payload.handoff::<ServerPayloadRegion>().into_bytes()",
        "payload.handoff::<DevicePayloadRegion>().into_bytes()",
        "TransportPayload::<ServerPayloadRegion>::from_bytes(bytes)",
    ] {
        assert!(
            source_all.contains(required),
            "process/server transport paths must retain payload handoff marker {required}"
        );
    }

    assert!(
        moirai_manifest.contains(
            "mnemosyne = [\"dep:mnemosyne\", \"moirai-core/mnemosyne\", \"moirai-executor/mnemosyne\"]"
        ),
        "top-level Moirai feature must retain Mnemosyne provider forwarding"
    );

    for prohibited in ["#[global_allocator]", "static ALLOC: mnemosyne::Mnemosyne"] {
        assert!(
            !moirai.contains(prohibited),
            "library crate must leave global allocator selection to the final binary: {prohibited}"
        );
    }
    for prohibited in ["Box<dyn PayloadRegion", "dyn PayloadRegion", ".clone().into_bytes()"] {
        assert!(
            !source_all.contains(prohibited),
            "payload handoff must not regress through {prohibited}"
        );
    }
}

#[test]
fn process_server_routed_execution_benchmark_uses_real_routes() {
    let manifest = read_benchmark("Cargo.toml");
    let source = read_benchmark("benches/process_server_routed_execution.rs");
    let checklist = read_benchmark("../docs/adr-008-checklist.md");

    assert!(
        manifest.contains("name = \"process_server_routed_execution\""),
        "benchmark manifest must register process_server_routed_execution"
    );
    assert!(
        manifest.contains("moirai-transport = { path = \"../moirai-transport\", features = [\"scheduler-routes\"] }"),
        "routed execution benchmark must compile route consumers"
    );

    for required in [
        "process_server_routed_execution",
        "run_server_route_once(0)",
        "run_process_route_once(0)",
        "let public_runtime = Moirai::new().expect(\"public facade runtime must build\")",
        "run_public_server_route_once(&public_runtime, 0)",
        "run_public_process_route_once(&public_runtime, 0)",
        "RoutedRemoteTaskClient::<ServerRoutePolicy>::new",
        "RoutedProcessTaskClient::<HybridRoutePolicy>::new",
        "black_box(&public_runtime)",
        "FixedRemoteTask::new",
        "RemoteCapabilityToken::<SumU64Capability>::new()",
        "execute_routed_server_task::<AsyncTask, ServerRoutePolicy, _, _>",
        "execute_routed_process_task::<AsyncTask, HybridRoutePolicy, _, _>",
        "RemoteTaskServer::new(server_address.clone())",
        "std::thread::spawn(move || server.serve_one().unwrap())",
        "ProcessSpec::new(std::env::current_exe().unwrap())",
        ".env(CHILD_MODE, \"1\")",
        "run_child_server()",
        "server.serve_one().unwrap()",
        "RemoteTaskOperation::SumU64(vec![13, 21, 34, iteration])",
        "RemoteTaskOperation::SumU64(vec![55, 89, 144, iteration])",
        "result.output",
        "output.result.output",
        "ProcessOutcome::Succeeded",
        "public_server_route_sum_u64",
        "public_process_route_sum_u64",
        "iter_custom",
        "without_plots",
    ] {
        assert!(
            source.contains(required),
            "routed execution benchmark must retain {required}"
        );
    }

    for prohibited in [
        "simulated",
        "simulation",
        "todo!",
        "unimplemented!",
        "Default::default()",
    ] {
        assert!(
            !source.contains(prohibited),
            "routed execution benchmark must not regress through {prohibited}"
        );
    }

    assert!(
        checklist.contains("End-to-end routed execution benchmark"),
        "ADR-008 checklist must retain routed execution benchmark tracking"
    );
}
