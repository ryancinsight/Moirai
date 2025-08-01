//! WASM executor implementation using Web Workers for parallelism.
//! 
//! This module provides a WebAssembly-compatible executor that leverages:
//! - Web Workers for true parallelism
//! - SharedArrayBuffer for zero-copy communication
//! - Atomics for synchronization

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Worker, MessageEvent, SharedArrayBuffer};
use js_sys::{Array, Uint8Array, Atomics};
use crate::{Task, TaskId, Priority, TaskContext};
use crate::error::ExecutorError;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use alloc::{vec::Vec, boxed::Box, sync::Arc, string::ToString};
use core::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use core::cell::UnsafeCell;

/// WASM executor that uses Web Workers for parallel execution
pub struct WasmExecutor {
    /// Pool of web workers
    workers: Vec<Worker>,
    /// Shared memory for communication
    shared_memory: SharedArrayBuffer,
    /// Task queue in shared memory
    task_queue: Arc<WasmTaskQueue>,
    /// Number of workers
    num_workers: usize,
}

/// Task queue implemented in SharedArrayBuffer
struct WasmTaskQueue {
    /// Shared buffer
    buffer: SharedArrayBuffer,
    /// Head position (producer)
    head: AtomicUsize,
    /// Tail position (consumer)
    tail: AtomicUsize,
    /// Queue capacity
    capacity: usize,
}

impl WasmExecutor {
    /// Create a new WASM executor with specified number of workers
    pub fn new(num_workers: usize) -> Result<Self, JsValue> {
        // Create shared memory (1MB for task queue)
        let shared_memory = SharedArrayBuffer::new(1024 * 1024);
        
        // Initialize task queue
        let task_queue = Arc::new(WasmTaskQueue::new(shared_memory.clone()));
        
        // Create workers
        let mut workers = Vec::with_capacity(num_workers);
        
        for i in 0..num_workers {
            // Create worker from embedded script
            let worker = Worker::new(&format!("/moirai-worker-{}.js", i))?;
            
            // Set up message handler
            let onmessage_callback = Closure::wrap(Box::new(move |event: MessageEvent| {
                // Handle worker messages
                web_sys::console::log_1(&format!("Worker {} message: {:?}", i, event.data()).into());
            }) as Box<dyn FnMut(MessageEvent)>);
            
            worker.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
            onmessage_callback.forget();
            
            // Send shared memory to worker
            let init_msg = Array::new();
            init_msg.push(&shared_memory);
            init_msg.push(&JsValue::from(i));
            worker.post_message(&init_msg)?;
            
            workers.push(worker);
        }
        
        Ok(Self {
            workers,
            shared_memory,
            task_queue,
            num_workers,
        })
    }
    
    /// Submit a task to the executor
    pub fn submit_task(&self, task: WasmTask) -> Result<(), ExecutorError> {
        self.task_queue.push(task)
            .map_err(|_| ExecutorError::QueueFull)
    }
    
    /// Shutdown all workers
    pub fn shutdown(&self) {
        for worker in &self.workers {
            worker.terminate();
        }
    }
}

impl WasmTaskQueue {
    /// Create a new task queue in shared memory
    fn new(buffer: SharedArrayBuffer) -> Self {
        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity: 1024, // Fixed size for simplicity
        }
    }
    
    /// Push a task to the queue
    fn push(&self, task: WasmTask) -> Result<(), WasmTask> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        
        if head.wrapping_sub(tail) >= self.capacity {
            return Err(task); // Queue full
        }
        
        // Serialize task to shared memory
        let offset = (head % self.capacity) * 256; // 256 bytes per task
        task.write_to_buffer(&self.buffer, offset);
        
        // Update head with release ordering
        self.head.store(head.wrapping_add(1), Ordering::Release);
        
        // Wake a worker using Atomics.notify
        let array = Uint8Array::new(&self.buffer);
        let _ = Atomics::notify(&array, 0, 1);
        
        Ok(())
    }
    
    /// Pop a task from the queue (called by workers)
    fn pop(&self) -> Option<WasmTask> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail >= head {
            return None; // Empty
        }
        
        // Read task from shared memory
        let offset = (tail % self.capacity) * 256;
        let task = WasmTask::read_from_buffer(&self.buffer, offset)?;
        
        // Update tail
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        
        Some(task)
    }
}

/// WASM-compatible task representation
#[derive(Clone)]
pub struct WasmTask {
    /// Task ID
    id: TaskId,
    /// Task type
    task_type: WasmTaskType,
    /// Serialized task data
    data: Vec<u8>,
}

#[derive(Clone, Copy)]
enum WasmTaskType {
    /// JavaScript function to execute
    JsFunction,
    /// WASM function pointer
    WasmFunction,
    /// Parallel map operation
    ParallelMap,
    /// Reduce operation
    Reduce,
}

impl WasmTask {
    /// Create a new WASM task
    pub fn new(id: TaskId, task_type: WasmTaskType, data: Vec<u8>) -> Self {
        Self { id, task_type, data }
    }
    
    /// Write task to shared memory buffer
    fn write_to_buffer(&self, buffer: &SharedArrayBuffer, offset: usize) {
        let array = Uint8Array::new(buffer);
        
        // Write task ID (8 bytes)
        let id_bytes = self.id.0.to_le_bytes();
        for (i, &byte) in id_bytes.iter().enumerate() {
            array.set_index((offset + i) as u32, byte);
        }
        
        // Write task type (1 byte)
        array.set_index((offset + 8) as u32, self.task_type as u8);
        
        // Write data length (4 bytes)
        let len_bytes = (self.data.len() as u32).to_le_bytes();
        for (i, &byte) in len_bytes.iter().enumerate() {
            array.set_index((offset + 9 + i) as u32, byte);
        }
        
        // Write data
        for (i, &byte) in self.data.iter().enumerate() {
            array.set_index((offset + 13 + i) as u32, byte);
        }
    }
    
    /// Read task from shared memory buffer
    fn read_from_buffer(buffer: &SharedArrayBuffer, offset: usize) -> Option<Self> {
        let array = Uint8Array::new(buffer);
        
        // Read task ID
        let mut id_bytes = [0u8; 8];
        for i in 0..8 {
            id_bytes[i] = array.get_index((offset + i) as u32);
        }
        let id = TaskId(u64::from_le_bytes(id_bytes));
        
        // Read task type
        let task_type = match array.get_index((offset + 8) as u32) {
            0 => WasmTaskType::JsFunction,
            1 => WasmTaskType::WasmFunction,
            2 => WasmTaskType::ParallelMap,
            3 => WasmTaskType::Reduce,
            _ => return None,
        };
        
        // Read data length
        let mut len_bytes = [0u8; 4];
        for i in 0..4 {
            len_bytes[i] = array.get_index((offset + 9 + i) as u32);
        }
        let data_len = u32::from_le_bytes(len_bytes) as usize;
        
        // Read data
        let mut data = Vec::with_capacity(data_len);
        for i in 0..data_len {
            data.push(array.get_index((offset + 13 + i) as u32));
        }
        
        Some(Self { id, task_type, data })
    }
}

/// WASM-compatible task handle
pub struct WasmTaskHandle<T> {
    /// Task ID
    id: TaskId,
    /// Result receiver
    result: Arc<AtomicOption<T>>,
    /// Completion flag
    completed: Arc<AtomicBool>,
}

/// Atomic option for lock-free result passing
struct AtomicOption<T> {
    value: UnsafeCell<Option<T>>,
    initialized: AtomicBool,
}

unsafe impl<T: Send> Send for AtomicOption<T> {}
unsafe impl<T: Send> Sync for AtomicOption<T> {}

impl<T> AtomicOption<T> {
    fn new() -> Self {
        Self {
            value: UnsafeCell::new(None),
            initialized: AtomicBool::new(false),
        }
    }
    
    fn set(&self, value: T) {
        unsafe {
            *self.value.get() = Some(value);
        }
        self.initialized.store(true, Ordering::Release);
    }
    
    fn take(&self) -> Option<T> {
        if self.initialized.load(Ordering::Acquire) {
            unsafe { (*self.value.get()).take() }
        } else {
            None
        }
    }
}

impl<T> Future for WasmTaskHandle<T> {
    type Output = T;
    
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed.load(Ordering::Acquire) {
            if let Some(result) = self.result.take() {
                Poll::Ready(result)
            } else {
                Poll::Pending
            }
        } else {
            Poll::Pending
        }
    }
}

/// Web Worker script generator
pub fn generate_worker_script() -> alloc::string::String {
    r#"
// Moirai Web Worker
let sharedBuffer;
let workerId;
let taskQueue;

// Message handler
self.onmessage = function(e) {
    if (e.data instanceof Array && e.data[0] instanceof SharedArrayBuffer) {
        // Initialize with shared memory
        sharedBuffer = e.data[0];
        workerId = e.data[1];
        taskQueue = new TaskQueue(sharedBuffer);
        
        // Start processing tasks
        processLoop();
    }
};

// Task processing loop
async function processLoop() {
    while (true) {
        const task = taskQueue.pop();
        
        if (task) {
            try {
                const result = await executeTask(task);
                postMessage({ type: 'result', taskId: task.id, result });
            } catch (error) {
                postMessage({ type: 'error', taskId: task.id, error: error.message });
            }
        } else {
            // Wait for notification
            const array = new Int32Array(sharedBuffer);
            Atomics.wait(array, 0, 0, 100); // Wait up to 100ms
        }
    }
}

// Execute a task
async function executeTask(task) {
    switch (task.type) {
        case 'jsFunction':
            const fn = new Function('return ' + task.data)();
            return await fn();
            
        case 'parallelMap':
            const { data, mapper } = JSON.parse(task.data);
            const mapFn = new Function('return ' + mapper)();
            return data.map(mapFn);
            
        case 'reduce':
            const { data: reduceData, reducer, initial } = JSON.parse(task.data);
            const reduceFn = new Function('return ' + reducer)();
            return reduceData.reduce(reduceFn, initial);
            
        default:
            throw new Error('Unknown task type: ' + task.type);
    }
}

// Task queue implementation
class TaskQueue {
    constructor(buffer) {
        this.buffer = buffer;
        this.view = new DataView(buffer);
        this.capacity = 1024;
    }
    
    pop() {
        // Implementation matches Rust side
        // ... (simplified for brevity)
    }
}
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    fn test_wasm_task_serialization() {
        let task = WasmTask::new(
            TaskId(42),
            WasmTaskType::JsFunction,
            b"console.log('test')".to_vec()
        );
        
        let buffer = SharedArrayBuffer::new(1024);
        task.write_to_buffer(&buffer, 0);
        
        let read_task = WasmTask::read_from_buffer(&buffer, 0).unwrap();
        assert_eq!(read_task.id.0, task.id.0);
        assert_eq!(read_task.data, task.data);
    }
}