//! Transport archive benchmark for borrowed receive views.
//!
//! The Moirai row uses `ArchiveView` to validate archive bytes and return an
//! `&str` borrowed from the message buffer. The reference row decodes the same
//! archive bytes into an owned `String`, so the benchmark isolates the receive
//! allocation avoided by rkyv-style archive views.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use moirai_transport::{
    payload::{DevicePayloadRegion, ThreadPayloadRegion, TransportPayload},
    safe_channel::{
        ArchiveSerialize, ArchivedMessage, ArchivedUniversalReceiver, ArchivedUniversalSender,
    },
    Address, TransportManager,
};
use std::{sync::Arc, time::Duration};

const SAMPLE_SIZE: usize = 20;
const MEASUREMENT_MILLIS: u64 = 500;
const WARM_UP_MILLIS: u64 = 100;
const PAYLOAD_REPETITIONS: usize = 8;
const PAYLOAD_UNIT: &str = "moirai-archive-view-payload:";

fn payload() -> String {
    PAYLOAD_UNIT.repeat(PAYLOAD_REPETITIONS)
}

fn verify_len(actual: &str, expected: &str) -> usize {
    assert_eq!(actual, expected);
    black_box(actual.len())
}

fn owned_decode_string(bytes: &[u8]) -> String {
    let len_bytes: [u8; 4] = bytes[0..4]
        .try_into()
        .expect("archive length prefix must exist");
    let len = u32::from_le_bytes(len_bytes) as usize;
    let payload = bytes
        .get(4..4 + len)
        .expect("archive payload must match length prefix");

    assert_eq!(bytes.len(), 4 + len);
    String::from_utf8(payload.to_owned()).expect("archive payload must be valid UTF-8")
}

fn borrowed_archive_view(message: &ArchivedMessage<String>, expected: &str) -> usize {
    let view = message
        .get()
        .expect("archive view must validate benchmark payload");

    verify_len(view, expected)
}

fn owned_decode_reference(bytes: &[u8], expected: &str) -> usize {
    let owned = owned_decode_string(bytes);

    verify_len(owned.as_str(), expected)
}

fn archived_transport_roundtrip(
    sender: &ArchivedUniversalSender<str>,
    receiver: &ArchivedUniversalReceiver<String>,
    expected: &str,
) -> usize {
    sender
        .send(expected)
        .expect("archived transport send must succeed");
    let message = receiver
        .recv()
        .expect("archived transport receive must succeed");
    let view = message
        .get()
        .expect("archived transport view must validate");

    verify_len(view, expected)
}

fn owned_transport_roundtrip(
    transport: &TransportManager,
    address: &Address,
    expected: &str,
) -> usize {
    transport
        .send(
            address,
            expected
                .archive_bytes()
                .expect("archive encode must succeed"),
        )
        .expect("raw transport send must succeed");
    let bytes = transport
        .recv(address)
        .expect("raw transport receive must succeed");
    let owned = owned_decode_string(&bytes);

    verify_len(owned.as_str(), expected)
}

fn device_region_handoff(
    payload: TransportPayload<ThreadPayloadRegion>,
    expected_bytes: &[u8],
) -> usize {
    let ptr = payload.as_bytes().as_ptr();
    let device_payload = payload.handoff::<DevicePayloadRegion>();

    assert_eq!(device_payload.as_bytes(), expected_bytes);
    assert_eq!(device_payload.as_bytes().as_ptr(), ptr);
    assert!(!TransportPayload::<DevicePayloadRegion>::pointer_transfer_allowed());
    black_box(device_payload.len())
}

fn bench_transport_archives(c: &mut Criterion) {
    let expected = payload();
    let archive_bytes = expected
        .archive_bytes()
        .expect("benchmark payload must encode");
    let archived_message = ArchivedMessage::<String>::from_bytes(archive_bytes.clone());

    let mut view_group = c.benchmark_group("transport_archive_view");
    view_group.sample_size(SAMPLE_SIZE);
    view_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    view_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    view_group.throughput(Throughput::Bytes(archive_bytes.len() as u64));

    view_group.bench_function("borrowed_archive_view", |bench| {
        bench.iter(|| borrowed_archive_view(black_box(&archived_message), black_box(&expected)));
    });

    view_group.bench_function("owned_decode_reference", |bench| {
        bench.iter(|| owned_decode_reference(black_box(&archive_bytes), black_box(&expected)));
    });

    view_group.finish();

    let transport = Arc::new(TransportManager::new());
    let archive_address = Address::Local("transport-archive-borrowed".to_string());
    let archived_sender =
        ArchivedUniversalSender::<str>::new(Arc::clone(&transport), archive_address.clone());
    let archived_receiver =
        ArchivedUniversalReceiver::<String>::new(Arc::clone(&transport), archive_address);

    let owned_transport = TransportManager::new();
    let owned_address = Address::Local("transport-archive-owned".to_string());

    let mut roundtrip_group = c.benchmark_group("transport_archive_roundtrip");
    roundtrip_group.sample_size(SAMPLE_SIZE);
    roundtrip_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    roundtrip_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    roundtrip_group.throughput(Throughput::Bytes(archive_bytes.len() as u64));

    roundtrip_group.bench_function("archived_transport_borrowed_view", |bench| {
        bench.iter(|| {
            archived_transport_roundtrip(
                black_box(&archived_sender),
                black_box(&archived_receiver),
                black_box(&expected),
            );
        });
    });

    roundtrip_group.bench_function("raw_transport_owned_decode_reference", |bench| {
        bench.iter(|| {
            owned_transport_roundtrip(
                black_box(&owned_transport),
                black_box(&owned_address),
                black_box(&expected),
            );
        });
    });

    roundtrip_group.finish();

    let mut handoff_group = c.benchmark_group("transport_payload_region_handoff");
    handoff_group.sample_size(SAMPLE_SIZE);
    handoff_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    handoff_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    handoff_group.throughput(Throughput::Bytes(archive_bytes.len() as u64));

    handoff_group.bench_function("device_region_owned_handoff", |bench| {
        bench.iter_batched(
            || TransportPayload::<ThreadPayloadRegion>::from_bytes(archive_bytes.clone()),
            |payload| device_region_handoff(payload, black_box(&archive_bytes)),
            BatchSize::SmallInput,
        );
    });

    handoff_group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = bench_transport_archives
}

criterion_main!(benches);
