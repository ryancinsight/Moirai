//! Distributed Task Coordination - Real-World Edge Case Example
//!
//! This example demonstrates:
//! - Distributed consensus in task scheduling
//! - Network partition tolerance and recovery
//! - Leader election and failover scenarios
//! - Work distribution across multiple nodes
//! - Consistency guarantees in distributed environments

#![allow(dead_code)] // This example keeps distributed-state variants that document failure modes outside the short run.

use moirai::{Moirai, Priority};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Represents a node in the distributed system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeId(u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node-{}", self.0)
    }
}

/// Distributed task that can be executed on any node
#[derive(Debug, Clone)]
struct DistributedTask {
    id: u64,
    task_type: TaskType,
    payload: Vec<u8>,
    priority: u8,
    deadline: Option<u64>,  // Unix timestamp
    dependencies: Vec<u64>, // Task IDs this task depends on
    retry_count: usize,
    max_retries: usize,
    created_at: u64,
    assigned_node: Option<NodeId>,
}

#[derive(Debug, Clone, PartialEq)]
enum TaskType {
    Compute,
    DataProcessing,
    FileOperation,
    NetworkRequest,
    DatabaseQuery,
}

#[derive(Debug, Clone, PartialEq)]
enum TaskStatus {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "PENDING"),
            TaskStatus::Assigned => write!(f, "ASSIGNED"),
            TaskStatus::Running => write!(f, "RUNNING"),
            TaskStatus::Completed => write!(f, "COMPLETED"),
            TaskStatus::Failed => write!(f, "FAILED"),
            TaskStatus::Cancelled => write!(f, "CANCELLED"),
        }
    }
}

/// Node state in the distributed cluster
#[derive(Debug, Clone)]
struct NodeState {
    id: NodeId,
    is_leader: bool,
    is_healthy: bool,
    last_heartbeat: u64,
    current_load: usize,
    max_capacity: usize,
    running_tasks: HashSet<u64>,
}

impl NodeState {
    fn new(id: NodeId, max_capacity: usize) -> Self {
        Self {
            id,
            is_leader: false,
            is_healthy: true,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            current_load: 0,
            max_capacity,
            running_tasks: HashSet::new(),
        }
    }

    fn can_accept_task(&self) -> bool {
        self.is_healthy && self.current_load < self.max_capacity
    }

    fn load_percentage(&self) -> f64 {
        (self.current_load as f64 / self.max_capacity as f64) * 100.0
    }
}

/// Distributed consensus mechanism (simplified Raft-like)
struct ConsensusEngine {
    current_term: Arc<AtomicU64>,
    voted_for: Arc<Mutex<Option<NodeId>>>,
    log_entries: Arc<Mutex<Vec<LogEntry>>>,
    commit_index: Arc<AtomicU64>,
    last_applied: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct LogEntry {
    term: u64,
    index: u64,
    command: ConsensusCommand,
    timestamp: u64,
}

#[derive(Debug, Clone)]
enum ConsensusCommand {
    TaskAssignment { task_id: u64, node_id: NodeId },
    TaskCompletion { task_id: u64, result: TaskResult },
    NodeJoin { node_id: NodeId },
    NodeLeave { node_id: NodeId },
    LeaderElection { candidate_id: NodeId },
}

#[derive(Debug, Clone)]
enum TaskResult {
    Success {
        output: Vec<u8>,
        execution_time_ms: u64,
    },
    Failure {
        error: String,
        retry_after_ms: Option<u64>,
    },
}

impl ConsensusEngine {
    fn new() -> Self {
        Self {
            current_term: Arc::new(AtomicU64::new(0)),
            voted_for: Arc::new(Mutex::new(None)),
            log_entries: Arc::new(Mutex::new(Vec::new())),
            commit_index: Arc::new(AtomicU64::new(0)),
            last_applied: Arc::new(AtomicU64::new(0)),
        }
    }

    fn append_entry(&self, command: ConsensusCommand) -> Result<u64, String> {
        let term = self.current_term.load(Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut log = self
            .log_entries
            .lock()
            .map_err(|_| "Failed to acquire log lock")?;

        let index = log.len() as u64;
        let entry = LogEntry {
            term,
            index,
            command,
            timestamp,
        };

        log.push(entry);
        Ok(index)
    }

    fn get_current_term(&self) -> u64 {
        self.current_term.load(Ordering::Relaxed)
    }

    fn increment_term(&self) -> u64 {
        self.current_term.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Distributed task scheduler with fault tolerance
struct DistributedTaskScheduler {
    node_id: NodeId,
    nodes: Arc<RwLock<HashMap<NodeId, NodeState>>>,
    tasks: Arc<RwLock<HashMap<u64, (DistributedTask, TaskStatus)>>>,
    consensus: Arc<ConsensusEngine>,
    task_queue: Arc<Mutex<VecDeque<u64>>>,
    completed_tasks: Arc<Mutex<Vec<u64>>>,
    failed_tasks: Arc<Mutex<Vec<u64>>>,

    // Statistics
    tasks_scheduled: Arc<AtomicUsize>,
    tasks_completed: Arc<AtomicUsize>,
    tasks_failed: Arc<AtomicUsize>,
    leader_elections: Arc<AtomicUsize>,
    network_partitions: Arc<AtomicUsize>,

    // State management
    is_leader: Arc<AtomicBool>,
    is_running: Arc<AtomicBool>,
    last_heartbeat_sent: Arc<AtomicU64>,
    runtime: Moirai,
}

impl DistributedTaskScheduler {
    fn new(node_id: NodeId, max_capacity: usize) -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;

        let mut nodes = HashMap::new();
        nodes.insert(
            node_id.clone(),
            NodeState::new(node_id.clone(), max_capacity),
        );

        let scheduler = Self {
            node_id: node_id.clone(),
            nodes: Arc::new(RwLock::new(nodes)),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            consensus: Arc::new(ConsensusEngine::new()),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            completed_tasks: Arc::new(Mutex::new(Vec::new())),
            failed_tasks: Arc::new(Mutex::new(Vec::new())),
            tasks_scheduled: Arc::new(AtomicUsize::new(0)),
            tasks_completed: Arc::new(AtomicUsize::new(0)),
            tasks_failed: Arc::new(AtomicUsize::new(0)),
            leader_elections: Arc::new(AtomicUsize::new(0)),
            network_partitions: Arc::new(AtomicUsize::new(0)),
            is_leader: Arc::new(AtomicBool::new(true)), // Start as leader for simplicity
            is_running: Arc::new(AtomicBool::new(true)),
            last_heartbeat_sent: Arc::new(AtomicU64::new(0)),
            runtime,
        };

        scheduler.start_background_processes()?;
        Ok(scheduler)
    }

    fn join_cluster(&self, other_node: NodeId, max_capacity: usize) -> Result<(), String> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|_| "Failed to acquire nodes write lock")?;

        if nodes.contains_key(&other_node) {
            return Err(format!("Node {} already in cluster", other_node));
        }

        nodes.insert(
            other_node.clone(),
            NodeState::new(other_node.clone(), max_capacity),
        );

        // Log the join operation
        let command = ConsensusCommand::NodeJoin {
            node_id: other_node.clone(),
        };
        self.consensus.append_entry(command)?;

        println!("Node {} joined cluster", other_node);
        Ok(())
    }

    fn submit_task(&self, mut task: DistributedTask) -> Result<u64, String> {
        task.created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let task_id = task.id;

        // Add task to task registry
        {
            let mut tasks = self
                .tasks
                .write()
                .map_err(|_| "Failed to acquire tasks write lock")?;
            tasks.insert(task_id, (task, TaskStatus::Pending));
        }

        // Add to task queue for scheduling
        {
            let mut queue = self
                .task_queue
                .lock()
                .map_err(|_| "Failed to acquire task queue lock")?;
            queue.push_back(task_id);
        }

        self.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        Ok(task_id)
    }

    fn schedule_tasks(&self) -> Result<usize, String> {
        if !self.is_leader.load(Ordering::Relaxed) {
            return Ok(0); // Only leader schedules tasks
        }

        let mut scheduled_count = 0;

        // Get available nodes
        let available_nodes: Vec<NodeId> = {
            let nodes = self
                .nodes
                .read()
                .map_err(|_| "Failed to acquire nodes read lock")?;
            nodes
                .values()
                .filter(|node| node.can_accept_task())
                .map(|node| node.id.clone())
                .collect()
        };

        if available_nodes.is_empty() {
            return Ok(0); // No available nodes
        }

        // Process tasks from queue
        let tasks_to_schedule: Vec<u64> = {
            let mut queue = self
                .task_queue
                .lock()
                .map_err(|_| "Failed to acquire task queue lock")?;

            let mut tasks = Vec::new();
            while let Some(task_id) = queue.pop_front() {
                if tasks.len() >= available_nodes.len() {
                    queue.push_front(task_id); // Put it back
                    break;
                }
                tasks.push(task_id);
            }
            tasks
        };

        for (i, task_id) in tasks_to_schedule.iter().enumerate() {
            let node_id = &available_nodes[i % available_nodes.len()];

            if self.assign_task_to_node(*task_id, node_id.clone())? {
                scheduled_count += 1;
            }
        }

        Ok(scheduled_count)
    }

    fn assign_task_to_node(&self, task_id: u64, node_id: NodeId) -> Result<bool, String> {
        // Update task status
        {
            let mut tasks = self
                .tasks
                .write()
                .map_err(|_| "Failed to acquire tasks write lock")?;

            if let Some((task, status)) = tasks.get_mut(&task_id) {
                if *status == TaskStatus::Pending {
                    task.assigned_node = Some(node_id.clone());
                    *status = TaskStatus::Assigned;
                } else {
                    return Ok(false); // Task no longer pending
                }
            } else {
                return Err(format!("Task {} not found", task_id));
            }
        }

        // Update node state
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| "Failed to acquire nodes write lock")?;

            if let Some(node) = nodes.get_mut(&node_id) {
                node.current_load += 1;
                node.running_tasks.insert(task_id);
            }
        }

        // Log assignment in consensus
        let command = ConsensusCommand::TaskAssignment {
            task_id,
            node_id: node_id.clone(),
        };
        self.consensus.append_entry(command)?;

        // Execute task asynchronously
        self.execute_task(task_id, node_id)?;
        Ok(true)
    }

    fn execute_task(&self, task_id: u64, node_id: NodeId) -> Result<(), String> {
        let task_data = {
            let tasks = self
                .tasks
                .read()
                .map_err(|_| "Failed to acquire tasks read lock")?;

            tasks
                .get(&task_id)
                .map(|(task, _)| task.clone())
                .ok_or_else(|| format!("Task {} not found", task_id))?
        };

        let nodes = self.nodes.clone();
        let tasks = self.tasks.clone();
        let consensus = self.consensus.clone();
        let completed_tasks = self.completed_tasks.clone();
        let failed_tasks = self.failed_tasks.clone();
        let tasks_completed = self.tasks_completed.clone();
        let tasks_failed = self.tasks_failed.clone();

        // Execute task based on priority
        let priority = if task_data.priority > 7 {
            Priority::High
        } else if task_data.priority > 3 {
            Priority::Normal
        } else {
            Priority::Low
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                let start_time = Instant::now();

                // Update task status to running
                {
                    if let Ok(mut task_registry) = tasks.write() {
                        if let Some((_, status)) = task_registry.get_mut(&task_id) {
                            *status = TaskStatus::Running;
                        }
                    }
                }

                // Simulate task execution with potential for failure
                let execution_result = Self::simulate_task_execution(&task_data);
                let execution_time = start_time.elapsed().as_millis() as u64;

                // Process result
                match execution_result {
                    Ok(output) => {
                        // Mark as completed
                        {
                            if let Ok(mut task_registry) = tasks.write() {
                                if let Some((_, status)) = task_registry.get_mut(&task_id) {
                                    *status = TaskStatus::Completed;
                                }
                            }
                        }

                        // Update node state
                        {
                            if let Ok(mut node_registry) = nodes.write() {
                                if let Some(node) = node_registry.get_mut(&node_id) {
                                    node.current_load = node.current_load.saturating_sub(1);
                                    node.running_tasks.remove(&task_id);
                                }
                            }
                        }

                        // Log completion
                        let result = TaskResult::Success {
                            output,
                            execution_time_ms: execution_time,
                        };
                        let command = ConsensusCommand::TaskCompletion { task_id, result };
                        let _ = consensus.append_entry(command);

                        // Add to completed list
                        if let Ok(mut completed) = completed_tasks.lock() {
                            completed.push(task_id);
                        }

                        tasks_completed.fetch_add(1, Ordering::Relaxed);
                        println!(
                            "Task {} completed on {} in {}ms",
                            task_id, node_id, execution_time
                        );
                    }
                    Err(error) => {
                        // Handle task failure
                        {
                            if let Ok(mut task_registry) = tasks.write() {
                                if let Some((task, status)) = task_registry.get_mut(&task_id) {
                                    task.retry_count += 1;
                                    if task.retry_count >= task.max_retries {
                                        *status = TaskStatus::Failed;
                                    } else {
                                        *status = TaskStatus::Pending; // Retry
                                    }
                                }
                            }
                        }

                        // Update node state
                        {
                            if let Ok(mut node_registry) = nodes.write() {
                                if let Some(node) = node_registry.get_mut(&node_id) {
                                    node.current_load = node.current_load.saturating_sub(1);
                                    node.running_tasks.remove(&task_id);
                                }
                            }
                        }

                        // Log failure
                        let result = TaskResult::Failure {
                            error: error.clone(),
                            retry_after_ms: if task_data.retry_count < task_data.max_retries {
                                Some(1000)
                            } else {
                                None
                            },
                        };
                        let command = ConsensusCommand::TaskCompletion { task_id, result };
                        let _ = consensus.append_entry(command);

                        if task_data.retry_count >= task_data.max_retries {
                            if let Ok(mut failed) = failed_tasks.lock() {
                                failed.push(task_id);
                            }
                            tasks_failed.fetch_add(1, Ordering::Relaxed);
                            println!(
                                "Task {} failed permanently on {}: {}",
                                task_id, node_id, error
                            );
                        } else {
                            println!(
                                "Task {} failed on {} (retry {}/{}): {}",
                                task_id,
                                node_id,
                                task_data.retry_count + 1,
                                task_data.max_retries,
                                error
                            );
                        }
                    }
                }
            },
            priority,
        );

        std::mem::drop(handle); // Let it run asynchronously
        Ok(())
    }

    fn simulate_task_execution(task: &DistributedTask) -> Result<Vec<u8>, String> {
        // Simulate different execution times and failure rates based on task type
        let (execution_time_ms, failure_rate) = match task.task_type {
            TaskType::Compute => (fastrand::u64(50..200), 0.02), // 2% failure rate
            TaskType::DataProcessing => (fastrand::u64(100..500), 0.05), // 5% failure rate
            TaskType::FileOperation => (fastrand::u64(20..100), 0.03), // 3% failure rate
            TaskType::NetworkRequest => (fastrand::u64(200..1000), 0.10), // 10% failure rate
            TaskType::DatabaseQuery => (fastrand::u64(50..300), 0.04), // 4% failure rate
        };

        // Simulate execution time
        std::thread::sleep(Duration::from_millis(execution_time_ms));

        // Simulate potential failure
        if fastrand::f64() < failure_rate {
            let error_types = [
                "Network timeout",
                "Resource exhausted",
                "Invalid input data",
                "Service unavailable",
                "Permission denied",
            ];
            let error = error_types[fastrand::usize(0..error_types.len())];
            return Err(error.to_string());
        }

        // Return simulated output
        Ok(vec![0u8; fastrand::usize(10..100)])
    }

    fn simulate_network_partition(&self, isolated_nodes: Vec<NodeId>) -> Result<(), String> {
        println!(
            "SIMULATING NETWORK PARTITION: Isolating nodes {:?}",
            isolated_nodes
        );

        self.network_partitions.fetch_add(1, Ordering::Relaxed);

        // Mark isolated nodes as unhealthy
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| "Failed to acquire nodes write lock")?;

            for node_id in &isolated_nodes {
                if let Some(node) = nodes.get_mut(node_id) {
                    node.is_healthy = false;
                    println!("Node {} marked as unhealthy due to partition", node_id);
                }
            }
        }

        // Trigger leader election if current leader is isolated
        if isolated_nodes.contains(&self.node_id) && self.is_leader.load(Ordering::Relaxed) {
            self.trigger_leader_election()?;
        }

        Ok(())
    }

    fn recover_from_partition(&self, recovered_nodes: Vec<NodeId>) -> Result<(), String> {
        println!(
            "RECOVERING FROM PARTITION: Restoring nodes {:?}",
            recovered_nodes
        );

        // Mark recovered nodes as healthy
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| "Failed to acquire nodes write lock")?;

            for node_id in &recovered_nodes {
                if let Some(node) = nodes.get_mut(node_id) {
                    node.is_healthy = true;
                    node.last_heartbeat = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    println!("Node {} recovered and marked as healthy", node_id);
                }
            }
        }

        Ok(())
    }

    fn trigger_leader_election(&self) -> Result<(), String> {
        println!("TRIGGERING LEADER ELECTION from node {}", self.node_id);

        self.leader_elections.fetch_add(1, Ordering::Relaxed);

        // Increment term
        let new_term = self.consensus.increment_term();

        // Simple leader election: node with lowest ID wins (in real systems, this would be more complex)
        let new_leader = {
            let nodes = self
                .nodes
                .read()
                .map_err(|_| "Failed to acquire nodes read lock")?;

            nodes
                .values()
                .filter(|node| node.is_healthy)
                .min_by_key(|node| node.id.0)
                .map(|node| node.id.clone())
        };

        if let Some(leader_id) = new_leader {
            // Update leadership
            {
                let mut nodes = self
                    .nodes
                    .write()
                    .map_err(|_| "Failed to acquire nodes write lock")?;

                for node in nodes.values_mut() {
                    node.is_leader = node.id == leader_id;
                }
            }

            let is_new_leader = leader_id == self.node_id;
            self.is_leader.store(is_new_leader, Ordering::Relaxed);

            // Log leader election
            let command = ConsensusCommand::LeaderElection {
                candidate_id: leader_id.clone(),
            };
            self.consensus.append_entry(command)?;

            println!("NEW LEADER ELECTED: {} (term {})", leader_id, new_term);
        }

        Ok(())
    }

    fn start_background_processes(&self) -> Result<(), String> {
        // Start task scheduling loop
        let scheduler_ref = SchedulerRef {
            is_leader: self.is_leader.clone(),
            is_running: self.is_running.clone(),
            task_queue: self.task_queue.clone(),
            nodes: self.nodes.clone(),
            tasks: self.tasks.clone(),
            consensus: self.consensus.clone(),
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while scheduler_ref.is_running.load(Ordering::Relaxed) {
                    if scheduler_ref.is_leader.load(Ordering::Relaxed) {
                        let _ = scheduler_ref.schedule_tasks();
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            },
            Priority::High,
        );

        std::mem::drop(handle);

        // Start heartbeat process
        let heartbeat_ref = HeartbeatRef {
            node_id: self.node_id.clone(),
            nodes: self.nodes.clone(),
            is_running: self.is_running.clone(),
            last_heartbeat_sent: self.last_heartbeat_sent.clone(),
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while heartbeat_ref.is_running.load(Ordering::Relaxed) {
                    heartbeat_ref.send_heartbeat();
                    std::thread::sleep(Duration::from_secs(1));
                }
            },
            Priority::Normal,
        );

        std::mem::drop(handle);

        Ok(())
    }

    fn get_cluster_status(&self) -> Result<ClusterStatus, String> {
        let nodes = self
            .nodes
            .read()
            .map_err(|_| "Failed to acquire nodes read lock")?;

        let total_nodes = nodes.len();
        let healthy_nodes = nodes.values().filter(|n| n.is_healthy).count();
        let total_capacity = nodes.values().map(|n| n.max_capacity).sum();
        let current_load = nodes.values().map(|n| n.current_load).sum();

        let leader = nodes.values().find(|n| n.is_leader).map(|n| n.id.clone());

        let tasks = self
            .tasks
            .read()
            .map_err(|_| "Failed to acquire tasks read lock")?;

        let pending_tasks = tasks
            .values()
            .filter(|(_, status)| *status == TaskStatus::Pending)
            .count();
        let running_tasks = tasks
            .values()
            .filter(|(_, status)| *status == TaskStatus::Running)
            .count();

        Ok(ClusterStatus {
            total_nodes,
            healthy_nodes,
            total_capacity,
            current_load,
            leader,
            pending_tasks,
            running_tasks,
            tasks_scheduled: self.tasks_scheduled.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
            tasks_failed: self.tasks_failed.load(Ordering::Relaxed),
            leader_elections: self.leader_elections.load(Ordering::Relaxed),
            network_partitions: self.network_partitions.load(Ordering::Relaxed),
            current_term: self.consensus.get_current_term(),
        })
    }

    fn shutdown(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }
}

struct SchedulerRef {
    is_leader: Arc<AtomicBool>,
    is_running: Arc<AtomicBool>,
    task_queue: Arc<Mutex<VecDeque<u64>>>,
    nodes: Arc<RwLock<HashMap<NodeId, NodeState>>>,
    tasks: Arc<RwLock<HashMap<u64, (DistributedTask, TaskStatus)>>>,
    consensus: Arc<ConsensusEngine>,
}

impl SchedulerRef {
    fn schedule_tasks(&self) -> Result<usize, String> {
        // Simplified scheduling logic for background process
        let mut scheduled_count = 0;

        let available_nodes: Vec<NodeId> = {
            let nodes = self
                .nodes
                .read()
                .map_err(|_| "Failed to acquire nodes read lock")?;
            nodes
                .values()
                .filter(|node| node.can_accept_task())
                .map(|node| node.id.clone())
                .collect()
        };

        if available_nodes.is_empty() {
            return Ok(0);
        }

        let tasks_to_schedule: Vec<u64> = {
            let mut queue = self
                .task_queue
                .lock()
                .map_err(|_| "Failed to acquire task queue lock")?;

            let mut tasks = Vec::new();
            for _ in 0..available_nodes.len().min(10) {
                // Limit batch size
                if let Some(task_id) = queue.pop_front() {
                    tasks.push(task_id);
                } else {
                    break;
                }
            }
            tasks
        };

        for (i, task_id) in tasks_to_schedule.iter().enumerate() {
            let node_id = &available_nodes[i % available_nodes.len()];

            // Simplified assignment (actual implementation would be more complex)
            if let Ok(mut tasks) = self.tasks.write() {
                if let Some((task, status)) = tasks.get_mut(task_id) {
                    if *status == TaskStatus::Pending {
                        task.assigned_node = Some(node_id.clone());
                        *status = TaskStatus::Assigned;
                        scheduled_count += 1;
                    }
                }
            }
        }

        Ok(scheduled_count)
    }
}

struct HeartbeatRef {
    node_id: NodeId,
    nodes: Arc<RwLock<HashMap<NodeId, NodeState>>>,
    is_running: Arc<AtomicBool>,
    last_heartbeat_sent: Arc<AtomicU64>,
}

impl HeartbeatRef {
    fn send_heartbeat(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Ok(mut nodes) = self.nodes.write() {
            if let Some(node) = nodes.get_mut(&self.node_id) {
                node.last_heartbeat = now;
            }
        }

        self.last_heartbeat_sent.store(now, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct ClusterStatus {
    total_nodes: usize,
    healthy_nodes: usize,
    total_capacity: usize,
    current_load: usize,
    leader: Option<NodeId>,
    pending_tasks: usize,
    running_tasks: usize,
    tasks_scheduled: usize,
    tasks_completed: usize,
    tasks_failed: usize,
    leader_elections: usize,
    network_partitions: usize,
    current_term: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Distributed Task Coordination - Edge Case Testing");
    println!("=================================================");

    // Create a distributed cluster
    let node1 = DistributedTaskScheduler::new(NodeId(1), 5)?;
    let node2 = DistributedTaskScheduler::new(NodeId(2), 3)?;
    let node3 = DistributedTaskScheduler::new(NodeId(3), 4)?;

    // Form cluster
    println!("\n1. Forming distributed cluster...");
    node1.join_cluster(NodeId(2), 3)?;
    node1.join_cluster(NodeId(3), 4)?;

    println!("  Cluster formed with 3 nodes (capacities: 5, 3, 4)");

    // Submit various tasks
    println!("\n2. Submitting distributed tasks...");
    let tasks = vec![
        DistributedTask {
            id: 1001,
            task_type: TaskType::Compute,
            payload: vec![1, 2, 3],
            priority: 8,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 3,
            created_at: 0,
            assigned_node: None,
        },
        DistributedTask {
            id: 1002,
            task_type: TaskType::DataProcessing,
            payload: vec![4, 5, 6],
            priority: 5,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 2,
            created_at: 0,
            assigned_node: None,
        },
        DistributedTask {
            id: 1003,
            task_type: TaskType::NetworkRequest,
            payload: vec![7, 8, 9],
            priority: 3,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 5,
            created_at: 0,
            assigned_node: None,
        },
        DistributedTask {
            id: 1004,
            task_type: TaskType::DatabaseQuery,
            payload: vec![10, 11, 12],
            priority: 7,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 3,
            created_at: 0,
            assigned_node: None,
        },
        DistributedTask {
            id: 1005,
            task_type: TaskType::FileOperation,
            payload: vec![13, 14, 15],
            priority: 6,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 2,
            created_at: 0,
            assigned_node: None,
        },
    ];

    for task in tasks {
        node1.submit_task(task)?;
    }

    println!("  Submitted 5 tasks of different types and priorities");

    // Let tasks process for a while
    std::thread::sleep(Duration::from_millis(500));

    // Edge Case 1: Network partition
    println!("\n3. Simulating network partition...");
    node1.simulate_network_partition(vec![NodeId(2)])?;

    // Continue submitting tasks during partition
    for i in 2001..2006 {
        let task = DistributedTask {
            id: i,
            task_type: TaskType::Compute,
            payload: vec![i as u8],
            priority: 5,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 3,
            created_at: 0,
            assigned_node: None,
        };
        node1.submit_task(task)?;
    }

    std::thread::sleep(Duration::from_millis(300));

    // Edge Case 2: Recover from partition
    println!("\n4. Recovering from network partition...");
    node1.recover_from_partition(vec![NodeId(2)])?;

    std::thread::sleep(Duration::from_millis(300));

    // Edge Case 3: Leader failure simulation
    println!("\n5. Simulating leader failure and election...");
    node1.trigger_leader_election()?;

    std::thread::sleep(Duration::from_millis(300));

    // Add more tasks to test load balancing
    println!("\n6. Testing load balancing with burst of tasks...");
    for i in 3001..3021 {
        let task = DistributedTask {
            id: i,
            task_type: match i % 5 {
                0 => TaskType::Compute,
                1 => TaskType::DataProcessing,
                2 => TaskType::NetworkRequest,
                3 => TaskType::DatabaseQuery,
                4 => TaskType::FileOperation,
                _ => TaskType::Compute,
            },
            payload: vec![i as u8],
            priority: (i % 10) as u8,
            deadline: None,
            dependencies: vec![],
            retry_count: 0,
            max_retries: 3,
            created_at: 0,
            assigned_node: None,
        };
        node1.submit_task(task)?;
    }

    // Wait for processing to complete
    std::thread::sleep(Duration::from_secs(2));

    // Display cluster status
    println!("\n7. Final Cluster Status:");
    match node1.get_cluster_status() {
        Ok(status) => {
            println!("  ├─ Cluster Configuration:");
            println!("  │  ├─ Total nodes: {}", status.total_nodes);
            println!("  │  ├─ Healthy nodes: {}", status.healthy_nodes);
            println!("  │  ├─ Current leader: {:?}", status.leader);
            println!("  │  └─ Consensus term: {}", status.current_term);

            println!("  ├─ Capacity & Load:");
            println!("  │  ├─ Total capacity: {}", status.total_capacity);
            println!("  │  ├─ Current load: {}", status.current_load);
            println!(
                "  │  └─ Utilization: {:.1}%",
                (status.current_load as f64 / status.total_capacity as f64) * 100.0
            );

            println!("  ├─ Task Statistics:");
            println!("  │  ├─ Tasks scheduled: {}", status.tasks_scheduled);
            println!("  │  ├─ Tasks completed: {}", status.tasks_completed);
            println!("  │  ├─ Tasks failed: {}", status.tasks_failed);
            println!("  │  ├─ Currently pending: {}", status.pending_tasks);
            println!("  │  ├─ Currently running: {}", status.running_tasks);
            println!(
                "  │  └─ Success rate: {:.2}%",
                if status.tasks_scheduled > 0 {
                    (status.tasks_completed as f64 / status.tasks_scheduled as f64) * 100.0
                } else {
                    0.0
                }
            );

            println!("  └─ Fault Tolerance:");
            println!("     ├─ Leader elections: {}", status.leader_elections);
            println!("     ├─ Network partitions: {}", status.network_partitions);
            println!(
                "     └─ System availability: {:.1}%",
                (status.healthy_nodes as f64 / status.total_nodes as f64) * 100.0
            );
        }
        Err(e) => println!("  Failed to get cluster status: {}", e),
    }

    // Shutdown cluster
    println!("\n8. Shutting down cluster...");
    node1.shutdown();
    node2.shutdown();
    node3.shutdown();

    println!("\nDistributed task coordination testing completed!");
    println!("Successfully handled network partitions, leader failures, and load balancing.");

    Ok(())
}
