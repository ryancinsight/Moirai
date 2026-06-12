//! Route identifier and count newtypes.

/// Scheduler worker-thread identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadId(usize);

impl ThreadId {
    /// Construct a thread identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Scheduler process identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessId(usize);

impl ProcessId {
    /// Construct a process identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Scheduler server identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerId(usize);

impl ServerId {
    /// Construct a server identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Async lane identifier inside a routed process.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncLaneId(usize);

impl AsyncLaneId {
    /// Construct an async lane identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Accelerator device identifier within one accelerator kind.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcceleratorId(usize);

impl AcceleratorId {
    /// Construct an accelerator identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Heterogeneous compute target family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorKind {
    /// CPU device placement metadata.
    Cpu,
    /// GPU device placement metadata.
    Gpu,
    /// TPU device placement metadata.
    Tpu,
    /// NPU device placement metadata.
    Npu,
}

impl AcceleratorKind {
    #[inline]
    pub(crate) const fn checksum_tag(self) -> usize {
        match self {
            Self::Cpu => 1,
            Self::Gpu => 2,
            Self::Tpu => 3,
            Self::Npu => 4,
        }
    }
}

/// Number of local scheduler worker threads.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerCount(usize);

impl WorkerCount {
    /// Construct a non-zero worker count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(if count == 0 { 1 } else { count })
    }

    /// Return the normalized count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of process route targets.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessCount(usize);

impl ProcessCount {
    /// Construct a non-zero process count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(if count == 0 { 1 } else { count })
    }

    /// Return the normalized count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of async lanes in each process target.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncLanesPerProcess(usize);

impl AsyncLanesPerProcess {
    /// Construct a non-zero async-lane count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(if count == 0 { 1 } else { count })
    }

    /// Return the normalized count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of server route targets.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerCount(usize);

impl ServerCount {
    /// Construct a server count. Zero disables server route targets.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(count)
    }

    /// Return the count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Available accelerator route counts by device family.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AcceleratorCounts {
    cpu: usize,
    gpu: usize,
    tpu: usize,
    npu: usize,
}

impl AcceleratorCounts {
    /// Construct accelerator counts by family.
    #[inline]
    pub const fn new(cpu: usize, gpu: usize, tpu: usize, npu: usize) -> Self {
        Self { cpu, gpu, tpu, npu }
    }

    /// Return the CPU placement count.
    #[inline]
    pub const fn cpu(self) -> usize {
        self.cpu
    }

    /// Return the GPU placement count.
    #[inline]
    pub const fn gpu(self) -> usize {
        self.gpu
    }

    /// Return the TPU placement count.
    #[inline]
    pub const fn tpu(self) -> usize {
        self.tpu
    }

    /// Return the NPU placement count.
    #[inline]
    pub const fn npu(self) -> usize {
        self.npu
    }

    /// Return the total accelerator target count.
    #[inline]
    pub const fn total(self) -> usize {
        self.cpu + self.gpu + self.tpu + self.npu
    }
}
