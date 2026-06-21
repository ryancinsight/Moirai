use std::sync::Arc;

/// Zero-copy message for efficient communication
pub struct Message<T> {
    /// The shared data
    data: Arc<T>,
}

impl<T> Message<T> {
    /// Create a new message
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Get a reference to the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Take ownership of the data if this is the only reference
    pub fn try_unwrap(self) -> Result<T, Self> {
        Arc::try_unwrap(self.data).map_err(|data| Self { data })
    }
}

impl<T> Clone for Message<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

impl<T> std::fmt::Debug for Message<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Message")
            .field("data", &format!("Arc<{}>", std::any::type_name::<T>()))
            .finish()
    }
}
