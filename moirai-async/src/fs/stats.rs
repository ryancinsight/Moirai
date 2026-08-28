/// Statistics for file operations
#[derive(Debug, Default, Clone)]
pub struct FileStats {
    /// Total bytes read through stateful stream operations on this handle.
    pub bytes_read: u64,
    /// Total bytes written through this handle.
    pub bytes_written: u64,
    /// Count of completed stateful stream read calls.
    pub read_operations: u64,
    /// Count of completed write calls.
    pub write_operations: u64,
    /// Count of completed seek calls.
    pub seek_operations: u64,
}
