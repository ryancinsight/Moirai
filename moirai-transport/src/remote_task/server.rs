use super::{execute_remote_task, RemoteTaskEnvelope, RemoteTaskId};
use crate::{
    payload::{archive_transport_payload, ServerPayloadRegion, TransportPayload},
    read_network_frame_from_stream,
    safe_channel::ArchivedMessage,
    Address, NetworkTransport, RemoteAddress, Transport, TransportError, TransportResult,
};
use std::{
    net::TcpListener,
    sync::{mpsc, Arc, Mutex},
    thread,
};

/// Single-request remote task server.
#[derive(Debug, Clone)]
pub struct RemoteTaskServer {
    bind: RemoteAddress,
}

impl RemoteTaskServer {
    /// Construct a remote task server bound to `bind`.
    pub fn new(bind: RemoteAddress) -> Self {
        Self { bind }
    }

    /// Receive, execute, and return one remote task.
    pub fn serve_one(&self) -> TransportResult<RemoteTaskId> {
        let bytes = NetworkTransport {}.recv(&Address::Remote(self.bind.clone()))?;
        let payload = TransportPayload::<ServerPayloadRegion>::from_bytes(bytes);
        let message = ArchivedMessage::<RemoteTaskEnvelope>::from_bytes(payload.into_bytes());
        let request = message.get()?;
        let result = execute_remote_task(&request);
        let task_id = result.task_id;

        let payload = archive_transport_payload::<ServerPayloadRegion, _>(&result)?;
        NetworkTransport {}.send(&Address::Remote(request.reply_to), payload.into_bytes())?;
        Ok(task_id)
    }
}

/// Bounded request queue capacity for a remote task server.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTaskQueueCapacity(usize);

impl RemoteTaskQueueCapacity {
    /// Construct a non-zero queue capacity.
    pub const fn new(capacity: usize) -> Self {
        Self(if capacity == 0 { 1 } else { capacity })
    }

    /// Return the normalized queue capacity.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Worker count for a bounded remote task server.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTaskWorkerCount(usize);

impl RemoteTaskWorkerCount {
    /// Construct a non-zero worker count.
    pub const fn new(workers: usize) -> Self {
        Self(if workers == 0 { 1 } else { workers })
    }

    /// Return the normalized worker count.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Request limit for one bounded server run.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTaskRequestLimit(usize);

impl RemoteTaskRequestLimit {
    /// Construct a request limit.
    pub const fn new(requests: usize) -> Self {
        Self(requests)
    }

    /// Return the request limit.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Bounded remote task server run summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTaskServerStats {
    /// Accepted requests.
    pub accepted: usize,
    /// Completed requests.
    pub completed: usize,
}

/// Remote task server with persistent listener ownership and bounded request buffering.
#[derive(Debug, Clone)]
pub struct BoundedRemoteTaskServer {
    bind: RemoteAddress,
    queue_capacity: RemoteTaskQueueCapacity,
    worker_count: RemoteTaskWorkerCount,
}

impl BoundedRemoteTaskServer {
    /// Construct a bounded remote task server.
    pub fn new(
        bind: RemoteAddress,
        queue_capacity: RemoteTaskQueueCapacity,
        worker_count: RemoteTaskWorkerCount,
    ) -> Self {
        Self {
            bind,
            queue_capacity,
            worker_count,
        }
    }

    /// Serve a bounded number of requests using one listener and one bounded queue.
    pub fn serve(&self, limit: RemoteTaskRequestLimit) -> TransportResult<RemoteTaskServerStats> {
        let listener =
            TcpListener::bind(remote_socket(&self.bind)).map_err(|_| TransportError::Closed)?;
        let (sender, receiver) = mpsc::sync_channel(self.queue_capacity.get());
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(self.worker_count.get());

        for _ in 0..self.worker_count.get() {
            let receiver = Arc::clone(&receiver);
            workers.push(thread::spawn(move || remote_task_worker(receiver)));
        }

        let mut accepted = 0usize;
        for _ in 0..limit.get() {
            let (mut stream, _) = listener.accept().map_err(|_| TransportError::Closed)?;
            let bytes = read_network_frame_from_stream(&mut stream)?;
            sender.send(bytes).map_err(|_| TransportError::Closed)?;
            accepted += 1;
        }
        drop(sender);

        let mut completed = 0usize;
        for worker in workers {
            completed += worker.join().map_err(|_| TransportError::Closed)??;
        }

        Ok(RemoteTaskServerStats {
            accepted,
            completed,
        })
    }
}

fn remote_task_worker(receiver: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>) -> TransportResult<usize> {
    let mut completed = 0usize;
    loop {
        let bytes = {
            let receiver = receiver.lock().map_err(|_| TransportError::Closed)?;
            receiver.recv()
        };
        let Ok(bytes) = bytes else {
            break;
        };
        let payload = TransportPayload::<ServerPayloadRegion>::from_bytes(bytes);
        let message = ArchivedMessage::<RemoteTaskEnvelope>::from_bytes(payload.into_bytes());
        let request = message.get()?;
        let result = execute_remote_task(&request);
        let payload = archive_transport_payload::<ServerPayloadRegion, _>(&result)?;
        NetworkTransport {}.send(&Address::Remote(request.reply_to), payload.into_bytes())?;
        completed += 1;
    }
    Ok(completed)
}

fn remote_socket(address: &RemoteAddress) -> String {
    format!("{}:{}", address.host, address.port)
}
