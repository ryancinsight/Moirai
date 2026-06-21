/// Statistics for file operations
#[derive(Debug, Default, Clone)]
pub struct FileStats {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub seek_operations: u64,
}
