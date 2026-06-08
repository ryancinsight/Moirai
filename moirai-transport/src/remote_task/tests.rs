use super::{
    BoundedRemoteTaskServer, RemoteTaskClient, RemoteTaskEnvelope, RemoteTaskId,
    RemoteTaskOperation, RemoteTaskOperationView, RemoteTaskOutput, RemoteTaskOutputView,
    RemoteTaskQueueCapacity, RemoteTaskRequestLimit, RemoteTaskResult, RemoteTaskServer,
    RemoteTaskWorkerCount,
};
use crate::safe_channel::{ArchiveSerialize, ArchiveView};
use crate::RemoteAddress;

#[test]
fn remote_task_envelope_view_borrows_echo_payload() {
    let payload = b"borrowed remote payload".to_vec();
    let envelope = RemoteTaskEnvelope {
        task_id: RemoteTaskId::new(7),
        reply_to: loopback_remote_address(),
        operation: RemoteTaskOperation::EchoBytes(payload.clone()),
    };
    let bytes = envelope.archive_bytes().unwrap();
    let view = RemoteTaskEnvelope::view_archive(&bytes).unwrap();

    match view.operation {
        RemoteTaskOperationView::EchoBytes(bytes) => assert_eq!(bytes, payload.as_slice()),
        RemoteTaskOperationView::SumU64(_) => panic!("expected echo operation"),
    }
}

#[test]
fn remote_task_envelope_view_sums_without_materialized_vector() {
    let values = vec![1u64, 2, u64::MAX, 9];
    let envelope = RemoteTaskEnvelope {
        task_id: RemoteTaskId::new(8),
        reply_to: loopback_remote_address(),
        operation: RemoteTaskOperation::SumU64(values),
    };
    let bytes = envelope.archive_bytes().unwrap();
    let view = RemoteTaskEnvelope::view_archive(&bytes).unwrap();

    match view.operation {
        RemoteTaskOperationView::SumU64(values) => {
            assert_eq!(values.len(), 4);
            assert_eq!(values.wrapping_sum(), 11);
        }
        RemoteTaskOperationView::EchoBytes(_) => panic!("expected sum operation"),
    }
}

#[test]
fn remote_task_result_view_borrows_bytes() {
    let result = RemoteTaskResult {
        task_id: RemoteTaskId::new(9),
        output: RemoteTaskOutput::Bytes(b"remote result".to_vec()),
    };
    let bytes = result.archive_bytes().unwrap();
    let view = RemoteTaskResult::view_archive(&bytes).unwrap();

    assert_eq!(view.task_id, result.task_id);
    match view.output {
        RemoteTaskOutputView::Bytes(bytes) => assert_eq!(bytes, b"remote result"),
        RemoteTaskOutputView::U64(_) => panic!("expected byte result"),
    }
}

#[test]
fn remote_task_client_server_executes_sum_roundtrip() {
    let server_address = loopback_remote_address();
    let reply_address = loopback_remote_address();
    let server = RemoteTaskServer::new(server_address.clone());
    let server_thread = std::thread::spawn(move || server.serve_one().unwrap());
    let client = RemoteTaskClient::new(server_address, reply_address);
    let task_id = RemoteTaskId::new(10);

    let result = client
        .execute(task_id, RemoteTaskOperation::SumU64(vec![5, 8, 13]))
        .unwrap();

    assert_eq!(server_thread.join().unwrap(), task_id);
    assert_eq!(result.output, RemoteTaskOutput::U64(26));
}

#[test]
fn remote_task_client_server_executes_echo_roundtrip() {
    let server_address = loopback_remote_address();
    let reply_address = loopback_remote_address();
    let server = RemoteTaskServer::new(server_address.clone());
    let server_thread = std::thread::spawn(move || server.serve_one().unwrap());
    let client = RemoteTaskClient::new(server_address, reply_address);
    let task_id = RemoteTaskId::new(11);

    let result = client
        .execute(
            task_id,
            RemoteTaskOperation::EchoBytes(b"echo route result".to_vec()),
        )
        .unwrap();

    assert_eq!(server_thread.join().unwrap(), task_id);
    assert_eq!(
        result.output,
        RemoteTaskOutput::Bytes(b"echo route result".to_vec())
    );
}

#[test]
fn bounded_remote_task_server_executes_multiple_requests_with_bounded_queue() {
    let server_address = loopback_remote_address();
    let first_reply = loopback_remote_address();
    let second_reply = loopback_remote_address();
    let server = BoundedRemoteTaskServer::new(
        server_address.clone(),
        RemoteTaskQueueCapacity::new(1),
        RemoteTaskWorkerCount::new(1),
    );
    let server_thread =
        std::thread::spawn(move || server.serve(RemoteTaskRequestLimit::new(2)).unwrap());

    let first = RemoteTaskClient::new(server_address.clone(), first_reply)
        .execute(
            RemoteTaskId::new(12),
            RemoteTaskOperation::SumU64(vec![34, 55]),
        )
        .unwrap();
    let second = RemoteTaskClient::new(server_address, second_reply)
        .execute(
            RemoteTaskId::new(13),
            RemoteTaskOperation::EchoBytes(b"bounded server".to_vec()),
        )
        .unwrap();
    let stats = server_thread.join().unwrap();

    assert_eq!(first.output, RemoteTaskOutput::U64(89));
    assert_eq!(
        second.output,
        RemoteTaskOutput::Bytes(b"bounded server".to_vec())
    );
    assert_eq!(stats.accepted, 2);
    assert_eq!(stats.completed, 2);
}

#[test]
fn remote_task_archives_reject_malformed_bytes() {
    assert!(RemoteTaskEnvelope::view_archive(&[1, 2, 3]).is_err());
    assert!(RemoteTaskResult::view_archive(&[1, 2, 3]).is_err());

    let result = RemoteTaskResult {
        task_id: RemoteTaskId::new(12),
        output: RemoteTaskOutput::U64(99),
    };
    let mut bytes = result.archive_bytes().unwrap();
    bytes.push(0);
    assert!(RemoteTaskResult::view_archive(&bytes).is_err());
}

fn loopback_remote_address() -> RemoteAddress {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    RemoteAddress {
        host: "127.0.0.1".to_string(),
        port,
        service: "moirai-remote-task".to_string(),
    }
}
