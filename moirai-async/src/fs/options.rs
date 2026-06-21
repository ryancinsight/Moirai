/// Configuration for file operations
#[derive(Debug, Clone)]
pub struct FileOpenOptions {
    /// Open for reading
    pub read: bool,
    /// Open for writing
    pub write: bool,
    /// Create if not exists
    pub create: bool,
    /// Append mode
    pub append: bool,
    /// Truncate existing content
    pub truncate: bool,
    /// File permissions (Unix only)
    pub mode: Option<u32>,
}

impl Default for FileOpenOptions {
    fn default() -> Self {
        Self {
            read: true,
            write: false,
            create: false,
            append: false,
            truncate: false,
            mode: None,
        }
    }
}

impl FileOpenOptions {
    /// Create options for read-only access
    pub fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            create: false,
            append: false,
            truncate: false,
            mode: None,
        }
    }

    /// Create options for write-only access (creates if not exists)
    pub fn write_only() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            append: false,
            truncate: true,
            mode: None,
        }
    }

    /// Create options for append access
    pub fn append_only() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            append: true,
            truncate: false,
            mode: None,
        }
    }

    /// Create options for read-write access
    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            create: true,
            append: false,
            truncate: false,
            mode: None,
        }
    }
}
