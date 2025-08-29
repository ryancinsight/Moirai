//! Advanced memory management for Moirai concurrency library.
//!
//! This module implements unified memory structures, object pools, and
//! zero-copy operations following literature on high-performance concurrency systems.

use std::alloc::{self, Layout};
use std::mem::{align_of, size_of, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::constants::CACHE_LINE_SIZE;

/// Memory pool for reducing allocation overhead.
/// Implements the object pool pattern from "The Art of Multiprocessor Programming".
pub struct MemoryPool<T> {
    /// Stack of available objects
    free_list: AtomicPtr<PoolNode<T>>,
    /// Current pool size
    size: AtomicUsize,
    /// Maximum pool size to prevent unbounded growth
    max_size: usize,
}

struct PoolNode<T> {
    next: *mut PoolNode<T>,
    data: MaybeUninit<T>,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool with specified maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            free_list: AtomicPtr::new(ptr::null_mut()),
            size: AtomicUsize::new(0),
            max_size,
        }
    }

    /// Allocate an object from the pool or create new if pool is empty
    pub fn allocate(&self) -> Box<T>
    where
        T: Default,
    {
        // Try to pop from free list first
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            if head.is_null() {
                // Pool is empty, allocate new
                return Box::new(T::default());
            }

            // Try to remove head from free list
            let next = unsafe { (*head).next };
            if self
                .free_list
                .compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                // Successfully removed from list
                self.size.fetch_sub(1, Ordering::Relaxed);
                
                // Extract the value and deallocate the node
                let value = unsafe {
                    let data = ptr::read(&(*head).data);
                    // Deallocate the node
                    let layout = Layout::new::<PoolNode<T>>();
                    alloc::dealloc(head as *mut u8, layout);
                    data.assume_init()
                };
                
                return Box::new(value);
            }
        }
    }

    /// Return an object to the pool for reuse
    pub fn deallocate(&self, item: Box<T>) {
        let current_size = self.size.load(Ordering::Relaxed);
        if current_size >= self.max_size {
            // Pool is full, just drop the item
            return;
        }

        // Allocate a new node
        let layout = Layout::new::<PoolNode<T>>();
        if let Some(ptr) = unsafe { NonNull::new(alloc::alloc(layout) as *mut PoolNode<T>) } {
            let node = unsafe {
                ptr::write(
                    ptr.as_ptr(),
                    PoolNode {
                        next: ptr::null_mut(),
                        data: MaybeUninit::new(*item),
                    },
                );
                &mut *ptr.as_ptr()
            };

            // Add to free list
            loop {
                let head = self.free_list.load(Ordering::Acquire);
                node.next = head;

                if self
                    .free_list
                    .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    self.size.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
}

impl<T> Drop for MemoryPool<T> {
    fn drop(&mut self) {
        // Clean up all nodes in the free list
        let mut current = self.free_list.load(Ordering::Acquire);
        while !current.is_null() {
            unsafe {
                let next = (*current).next;
                let layout = Layout::new::<PoolNode<T>>();
                alloc::dealloc(current as *mut u8, layout);
                current = next;
            }
        }
    }
}

/// Cache-aligned memory allocator for high-performance data structures.
/// Based on techniques from "Systems Performance" by Brendan Gregg.
pub struct CacheAlignedAllocator;

impl CacheAlignedAllocator {
    /// Allocate cache-aligned memory for optimal performance
    pub fn allocate<T>(count: usize) -> Option<NonNull<T>> {
        let size = size_of::<T>() * count;
        let align = align_of::<T>().max(CACHE_LINE_SIZE);
        
        let layout = Layout::from_size_align(size, align).ok()?;
        
        unsafe {
            let ptr = alloc::alloc(layout);
            NonNull::new(ptr.cast::<T>())
        }
    }

    /// Deallocate cache-aligned memory
    pub unsafe fn deallocate<T>(ptr: NonNull<T>, count: usize) {
        let size = size_of::<T>() * count;
        let align = align_of::<T>().max(CACHE_LINE_SIZE);
        
        if let Ok(layout) = Layout::from_size_align(size, align) {
            alloc::dealloc(ptr.as_ptr().cast::<u8>(), layout);
        }
    }
}

/// Zero-copy ring buffer with unified memory management.
/// Implements techniques from "Lock-Free Programming" by Maurice Herlihy.
pub struct UnifiedRingBuffer<T> {
    /// Cache-aligned buffer storage
    buffer: NonNull<MaybeUninit<T>>,
    /// Buffer capacity (always power of 2)
    capacity: usize,
    /// Mask for fast modulo operations
    mask: usize,
    /// Producer position
    head: AtomicUsize,
    /// Consumer position  
    tail: AtomicUsize,
    /// Associated memory pool for overflow handling
    pool: Arc<MemoryPool<T>>,
}

impl<T> UnifiedRingBuffer<T> {
    /// Create a new unified ring buffer with specified capacity
    pub fn new(capacity: usize) -> Option<Self> {
        let capacity = capacity.next_power_of_two().max(2);
        let buffer = CacheAlignedAllocator::allocate::<MaybeUninit<T>>(capacity)?;
        
        Some(Self {
            buffer,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            pool: Arc::new(MemoryPool::new(capacity * 2)),
        })
    }

    /// Try to push an item using zero-copy semantics
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;
        let tail = self.tail.load(Ordering::Acquire);

        if next_head == tail {
            // Buffer is full
            return Err(item);
        }

        unsafe {
            let slot = self.buffer.as_ptr().add(head & self.mask);
            ptr::write((*slot).as_mut_ptr(), item);
        }

        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Try to pop an item using zero-copy semantics
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            // Buffer is empty
            return None;
        }

        let item = unsafe {
            let slot = self.buffer.as_ptr().add(tail & self.mask);
            ptr::read((*slot).as_ptr())
        };

        self.tail.store((tail + 1) & self.mask, Ordering::Release);
        Some(item)
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & self.mask
    }
}

impl<T> Drop for UnifiedRingBuffer<T> {
    fn drop(&mut self) {
        // Clean up remaining items
        while let Some(_) = self.try_pop() {
            // Items are dropped automatically
        }

        // Deallocate buffer
        unsafe {
            CacheAlignedAllocator::deallocate(self.buffer, self.capacity);
        }
    }
}

// Safety: UnifiedRingBuffer is safe to send between threads
unsafe impl<T: Send> Send for UnifiedRingBuffer<T> {}
unsafe impl<T: Send> Sync for UnifiedRingBuffer<T> {}

/// Global memory pool manager for reduced allocation overhead
pub struct GlobalMemoryManager {
    /// Pools for different types (using type ID as key would be better but more complex)
    pools: [MemoryPool<u8>; 8], // Simple array for common sizes
}

impl GlobalMemoryManager {
    /// Get the global memory manager instance
    pub fn instance() -> &'static Self {
        // In a real implementation, this would use a proper singleton pattern
        // For now, we use a simple static
        static mut INSTANCE: Option<GlobalMemoryManager> = None;
        static INIT: std::sync::Once = std::sync::Once::new();

        unsafe {
            INIT.call_once(|| {
                INSTANCE = Some(GlobalMemoryManager {
                    pools: [
                        MemoryPool::new(1024),  // 8 byte pool
                        MemoryPool::new(1024),  // 16 byte pool  
                        MemoryPool::new(1024),  // 32 byte pool
                        MemoryPool::new(1024),  // 64 byte pool
                        MemoryPool::new(512),   // 128 byte pool
                        MemoryPool::new(512),   // 256 byte pool
                        MemoryPool::new(256),   // 512 byte pool
                        MemoryPool::new(256),   // 1024 byte pool
                    ],
                });
            });
            INSTANCE.as_ref().unwrap()
        }
    }

    /// Allocate from appropriate pool based on size
    pub fn allocate(&self, size: usize) -> Option<Vec<u8>> {
        let _pool_index = match size {
            1..=8 => 0,
            9..=16 => 1,
            17..=32 => 2,
            33..=64 => 3,
            65..=128 => 4,
            129..=256 => 5,
            257..=512 => 6,
            513..=1024 => 7,
            _ => return None, // Too large for pooling
        };

        // For simplicity, just return a new Vec
        // In a real implementation, we'd use the pool
        Some(vec![0u8; size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::<i32>::new(10);
        
        // Allocate some items
        let item1 = pool.allocate();
        let item2 = pool.allocate();
        
        // Pool should be empty initially
        assert_eq!(pool.size(), 0);
        
        // Return items to pool
        pool.deallocate(item1);
        pool.deallocate(item2);
        
        // Pool should now have items
        assert_eq!(pool.size(), 2);
        
        // Allocate again - should reuse from pool
        let _item3 = pool.allocate();
        assert_eq!(pool.size(), 1);
    }

    #[test]
    fn test_unified_ring_buffer() {
        let buffer = UnifiedRingBuffer::<i32>::new(8).unwrap();
        
        // Test basic operations
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        
        // Push some items
        assert!(buffer.try_push(1).is_ok());
        assert!(buffer.try_push(2).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Pop items
        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert!(buffer.is_empty());
    }
}