//! Remote task envelopes over transport-owned bytes.

use crate::{
    payload::{archive_transport_payload, ServerPayloadRegion, TransportPayload},
    safe_channel::{ArchiveSerialize, ArchiveView, ArchivedMessage},
    Address, NetworkTransport, RemoteAddress, Transport, TransportError, TransportResult,
};

mod capability;
mod server;

pub use capability::{
    build_remote_operation, EchoBytesCapability, IntoRemoteOperation, RemoteCapability,
    RemoteCapabilityToken, RemoteTaskOperationKind, SumU64Capability,
};
pub use server::{
    BoundedRemoteTaskServer, RemoteTaskQueueCapacity, RemoteTaskRequestLimit, RemoteTaskServer,
    RemoteTaskServerStats, RemoteTaskWorkerCount,
};

const OP_ECHO_BYTES: u8 = 1;
const OP_SUM_U64: u8 = 2;
const RESULT_BYTES: u8 = 1;
const RESULT_U64: u8 = 2;

/// Remote task identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemoteTaskId(u64);

impl RemoteTaskId {
    /// Construct a remote task id.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw id.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Built-in remote task operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteTaskOperation {
    /// Return the same byte payload.
    EchoBytes(Vec<u8>),
    /// Sum `u64` values with wrapping arithmetic.
    SumU64(Vec<u64>),
}

/// Owned remote task envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteTaskEnvelope {
    /// Task id copied into the result envelope.
    pub task_id: RemoteTaskId,
    /// Address where the server sends the result.
    pub reply_to: RemoteAddress,
    /// Operation to execute.
    pub operation: RemoteTaskOperation,
}

/// Borrowed remote task operation view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteTaskOperationView<'a> {
    /// Borrowed byte payload.
    EchoBytes(&'a [u8]),
    /// Borrowed little-endian u64 list.
    SumU64(RemoteU64List<'a>),
}

/// Borrowed `u64` list backed by archive bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteU64List<'a> {
    bytes: &'a [u8],
    len: usize,
}

impl RemoteU64List<'_> {
    /// Number of `u64` values in the list.
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether the list is empty.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Compute the wrapping sum without materializing an owned vector.
    pub fn wrapping_sum(self) -> u64 {
        self.bytes
            .chunks_exact(core::mem::size_of::<u64>())
            .fold(0u64, |sum, chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("chunk size is fixed");
                sum.wrapping_add(u64::from_le_bytes(bytes))
            })
    }
}

/// Borrowed remote task envelope view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteTaskEnvelopeView<'a> {
    /// Task id copied into the result envelope.
    pub task_id: RemoteTaskId,
    /// Address where the server sends the result.
    pub reply_to: RemoteAddress,
    /// Borrowed operation view.
    pub operation: RemoteTaskOperationView<'a>,
}

/// Built-in remote task result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteTaskOutput {
    /// Returned byte payload.
    Bytes(Vec<u8>),
    /// Returned `u64` value.
    U64(u64),
}

/// Owned remote task result envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteTaskResult {
    /// Completed task id.
    pub task_id: RemoteTaskId,
    /// Operation output.
    pub output: RemoteTaskOutput,
}

/// Borrowed remote task result output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteTaskOutputView<'a> {
    /// Borrowed result bytes.
    Bytes(&'a [u8]),
    /// Result integer.
    U64(u64),
}

/// Borrowed remote task result envelope view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTaskResultView<'a> {
    /// Completed task id.
    pub task_id: RemoteTaskId,
    /// Borrowed operation output.
    pub output: RemoteTaskOutputView<'a>,
}

/// Remote task client for a fixed server and reply endpoint.
#[derive(Debug, Clone)]
pub struct RemoteTaskClient {
    server: RemoteAddress,
    reply_to: RemoteAddress,
}

impl RemoteTaskClient {
    /// Construct a remote task client.
    pub fn new(server: RemoteAddress, reply_to: RemoteAddress) -> Self {
        Self { server, reply_to }
    }

    /// Send a task, wait for its result, and validate the returned id.
    pub fn execute(
        &self,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> TransportResult<RemoteTaskResult> {
        let envelope = RemoteTaskEnvelope {
            task_id,
            reply_to: self.reply_to.clone(),
            operation,
        };

        let payload = archive_transport_payload::<ServerPayloadRegion, _>(&envelope)?;
        NetworkTransport {}.send(&Address::Remote(self.server.clone()), payload.into_bytes())?;
        let bytes = NetworkTransport {}.recv(&Address::Remote(self.reply_to.clone()))?;
        let payload = TransportPayload::<ServerPayloadRegion>::from_bytes(bytes);
        let message = ArchivedMessage::<RemoteTaskResult>::from_bytes(payload.into_bytes());
        let view = message.get()?;
        if view.task_id != task_id {
            return Err(TransportError::Closed);
        }

        Ok(view.into_owned())
    }
}

impl ArchiveSerialize for RemoteTaskEnvelope {
    fn archive_size_hint(&self) -> usize {
        core::mem::size_of::<u64>()
            + remote_address_size(&self.reply_to)
            + operation_size(&self.operation)
    }

    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()> {
        output.extend_from_slice(&self.task_id.get().to_le_bytes());
        encode_remote_address(&self.reply_to, output)?;
        match &self.operation {
            RemoteTaskOperation::EchoBytes(bytes) => {
                output.push(OP_ECHO_BYTES);
                encode_len_prefixed_bytes(bytes, output)?;
            }
            RemoteTaskOperation::SumU64(values) => {
                output.push(OP_SUM_U64);
                let len = u32::try_from(values.len()).map_err(|_| TransportError::Closed)?;
                output.extend_from_slice(&len.to_le_bytes());
                for value in values {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(())
    }
}

impl ArchiveView for RemoteTaskEnvelope {
    type Archived<'a> = RemoteTaskEnvelopeView<'a>;

    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>> {
        let mut cursor = ByteCursor::new(bytes);
        let task_id = RemoteTaskId::new(cursor.read_u64()?);
        let reply_to = cursor.read_remote_address()?;
        let op = cursor.read_u8()?;
        let operation = match op {
            OP_ECHO_BYTES => RemoteTaskOperationView::EchoBytes(cursor.read_len_prefixed_bytes()?),
            OP_SUM_U64 => {
                let len =
                    usize::try_from(cursor.read_u32()?).map_err(|_| TransportError::Closed)?;
                let byte_len = len
                    .checked_mul(core::mem::size_of::<u64>())
                    .ok_or(TransportError::Closed)?;
                let payload = cursor.read_exact(byte_len)?;
                RemoteTaskOperationView::SumU64(RemoteU64List {
                    bytes: payload,
                    len,
                })
            }
            _ => return Err(TransportError::Closed),
        };
        cursor.finish()?;

        Ok(RemoteTaskEnvelopeView {
            task_id,
            reply_to,
            operation,
        })
    }
}

impl ArchiveSerialize for RemoteTaskResult {
    fn archive_size_hint(&self) -> usize {
        core::mem::size_of::<u64>()
            + 1
            + match &self.output {
                RemoteTaskOutput::Bytes(bytes) => core::mem::size_of::<u32>() + bytes.len(),
                RemoteTaskOutput::U64(_) => core::mem::size_of::<u64>(),
            }
    }

    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()> {
        output.extend_from_slice(&self.task_id.get().to_le_bytes());
        match &self.output {
            RemoteTaskOutput::Bytes(bytes) => {
                output.push(RESULT_BYTES);
                encode_len_prefixed_bytes(bytes, output)?;
            }
            RemoteTaskOutput::U64(value) => {
                output.push(RESULT_U64);
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    }
}

impl ArchiveView for RemoteTaskResult {
    type Archived<'a> = RemoteTaskResultView<'a>;

    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>> {
        let mut cursor = ByteCursor::new(bytes);
        let task_id = RemoteTaskId::new(cursor.read_u64()?);
        let tag = cursor.read_u8()?;
        let output = match tag {
            RESULT_BYTES => RemoteTaskOutputView::Bytes(cursor.read_len_prefixed_bytes()?),
            RESULT_U64 => RemoteTaskOutputView::U64(cursor.read_u64()?),
            _ => return Err(TransportError::Closed),
        };
        cursor.finish()?;

        Ok(RemoteTaskResultView { task_id, output })
    }
}

impl RemoteTaskResultView<'_> {
    /// Materialize the borrowed result view into an owned result.
    pub fn into_owned(self) -> RemoteTaskResult {
        RemoteTaskResult {
            task_id: self.task_id,
            output: match self.output {
                RemoteTaskOutputView::Bytes(bytes) => RemoteTaskOutput::Bytes(bytes.to_vec()),
                RemoteTaskOutputView::U64(value) => RemoteTaskOutput::U64(value),
            },
        }
    }
}

pub(super) fn execute_remote_task(request: &RemoteTaskEnvelopeView<'_>) -> RemoteTaskResult {
    let output = match request.operation {
        RemoteTaskOperationView::EchoBytes(bytes) => RemoteTaskOutput::Bytes(bytes.to_vec()),
        RemoteTaskOperationView::SumU64(values) => RemoteTaskOutput::U64(values.wrapping_sum()),
    };

    RemoteTaskResult {
        task_id: request.task_id,
        output,
    }
}

fn encode_remote_address(address: &RemoteAddress, output: &mut Vec<u8>) -> TransportResult<()> {
    encode_len_prefixed_str(&address.host, output)?;
    output.extend_from_slice(&address.port.to_le_bytes());
    encode_len_prefixed_str(&address.service, output)
}

fn remote_address_size(address: &RemoteAddress) -> usize {
    core::mem::size_of::<u32>()
        + address.host.len()
        + core::mem::size_of::<u16>()
        + core::mem::size_of::<u32>()
        + address.service.len()
}

fn operation_size(operation: &RemoteTaskOperation) -> usize {
    1 + match operation {
        RemoteTaskOperation::EchoBytes(bytes) => core::mem::size_of::<u32>() + bytes.len(),
        RemoteTaskOperation::SumU64(values) => {
            core::mem::size_of::<u32>() + values.len() * core::mem::size_of::<u64>()
        }
    }
}

fn encode_len_prefixed_str(value: &str, output: &mut Vec<u8>) -> TransportResult<()> {
    encode_len_prefixed_bytes(value.as_bytes(), output)
}

fn encode_len_prefixed_bytes(value: &[u8], output: &mut Vec<u8>) -> TransportResult<()> {
    let len = u32::try_from(value.len()).map_err(|_| TransportError::Closed)?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(value);
    Ok(())
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_u8(&mut self) -> TransportResult<u8> {
        let value = *self.bytes.get(self.offset).ok_or(TransportError::Closed)?;
        self.offset += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> TransportResult<u16> {
        let bytes: [u8; 2] = self
            .read_exact(core::mem::size_of::<u16>())?
            .try_into()
            .map_err(|_| TransportError::Closed)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> TransportResult<u32> {
        let bytes: [u8; 4] = self
            .read_exact(core::mem::size_of::<u32>())?
            .try_into()
            .map_err(|_| TransportError::Closed)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> TransportResult<u64> {
        let bytes: [u8; 8] = self
            .read_exact(core::mem::size_of::<u64>())?
            .try_into()
            .map_err(|_| TransportError::Closed)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_len_prefixed_bytes(&mut self) -> TransportResult<&'a [u8]> {
        let len = usize::try_from(self.read_u32()?).map_err(|_| TransportError::Closed)?;
        self.read_exact(len)
    }

    fn read_len_prefixed_string(&mut self) -> TransportResult<String> {
        let bytes = self.read_len_prefixed_bytes()?;
        let value = core::str::from_utf8(bytes).map_err(|_| TransportError::Closed)?;
        Ok(value.to_string())
    }

    fn read_remote_address(&mut self) -> TransportResult<RemoteAddress> {
        let host = self.read_len_prefixed_string()?;
        let port = self.read_u16()?;
        let service = self.read_len_prefixed_string()?;
        Ok(RemoteAddress {
            host,
            port,
            service,
        })
    }

    fn read_exact(&mut self, len: usize) -> TransportResult<&'a [u8]> {
        let end = self.offset.checked_add(len).ok_or(TransportError::Closed)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(TransportError::Closed)?;
        self.offset = end;
        Ok(bytes)
    }

    fn finish(self) -> TransportResult<()> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(TransportError::Closed)
        }
    }
}

#[cfg(test)]
mod tests;
