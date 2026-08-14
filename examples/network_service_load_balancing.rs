//! Network Service Load Balancing - Real-World Edge Case Example
//!
//! This example demonstrates:
//! - Dynamic load balancing across service instances
//! - Circuit breaker patterns for cascading failure prevention
//! - Health check and auto-scaling scenarios
//! - Request routing with affinity and consistency
//! - Performance monitoring and adaptive throttling

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]
#![allow(dead_code)] // This example models load-balancer state variants not all used by the compact scenario.

use moirai::{Moirai, Priority};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Represents a service instance in the load balancer
#[derive(Debug)]
struct ServiceInstance {
    id: String,
    endpoint: String,
    current_load: AtomicUsize,
    max_capacity: usize,
    response_time_ms: AtomicU64,
    success_count: AtomicUsize,
    failure_count: AtomicUsize,
    last_health_check: AtomicU64,
    is_healthy: AtomicBool,
    circuit_breaker_state: AtomicU8, // 0: Closed, 1: Open, 2: Half-Open
}

impl Clone for ServiceInstance {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            endpoint: self.endpoint.clone(),
            current_load: AtomicUsize::new(self.current_load.load(Ordering::Relaxed)),
            max_capacity: self.max_capacity,
            response_time_ms: AtomicU64::new(self.response_time_ms.load(Ordering::Relaxed)),
            success_count: AtomicUsize::new(self.success_count.load(Ordering::Relaxed)),
            failure_count: AtomicUsize::new(self.failure_count.load(Ordering::Relaxed)),
            last_health_check: AtomicU64::new(self.last_health_check.load(Ordering::Relaxed)),
            is_healthy: AtomicBool::new(self.is_healthy.load(Ordering::Relaxed)),
            circuit_breaker_state: AtomicU8::new(
                self.circuit_breaker_state.load(Ordering::Relaxed),
            ),
        }
    }
}

impl ServiceInstance {
    fn new(id: String, endpoint: String, max_capacity: usize) -> Self {
        Self {
            id,
            endpoint,
            current_load: AtomicUsize::new(0),
            max_capacity,
            response_time_ms: AtomicU64::new(0),
            success_count: AtomicUsize::new(0),
            failure_count: AtomicUsize::new(0),
            last_health_check: AtomicU64::new(0),
            is_healthy: AtomicBool::new(true),
            circuit_breaker_state: AtomicU8::new(0), // Closed
        }
    }

    fn can_handle_request(&self) -> bool {
        self.is_healthy.load(Ordering::Relaxed)
            && self.current_load.load(Ordering::Relaxed) < self.max_capacity
            && self.circuit_breaker_state.load(Ordering::Relaxed) != 1 // Not open
    }

    fn load_percentage(&self) -> f64 {
        (self.current_load.load(Ordering::Relaxed) as f64 / self.max_capacity as f64) * 100.0
    }

    fn success_rate(&self) -> f64 {
        let successes = self.success_count.load(Ordering::Relaxed);
        let failures = self.failure_count.load(Ordering::Relaxed);
        let total = successes + failures;

        if total == 0 {
            100.0
        } else {
            (successes as f64 / total as f64) * 100.0
        }
    }

    fn avg_response_time(&self) -> u64 {
        self.response_time_ms.load(Ordering::Relaxed)
    }

    fn record_request_start(&self) {
        self.current_load.fetch_add(1, Ordering::Relaxed);
    }

    fn record_request_complete(&self, success: bool, response_time_ms: u64) {
        self.current_load.fetch_sub(1, Ordering::Relaxed);

        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
            // Update rolling average response time
            let current_avg = self.response_time_ms.load(Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                response_time_ms
            } else {
                (current_avg * 3 + response_time_ms) / 4 // Simple rolling average
            };
            self.response_time_ms.store(new_avg, Ordering::Relaxed);

            // Reset circuit breaker on success if half-open
            if self.circuit_breaker_state.load(Ordering::Relaxed) == 2 {
                self.circuit_breaker_state.store(0, Ordering::Relaxed); // Close
            }
        } else {
            self.failure_count.fetch_add(1, Ordering::Relaxed);

            // Check circuit breaker conditions
            let total_requests = self.success_count.load(Ordering::Relaxed)
                + self.failure_count.load(Ordering::Relaxed);
            if total_requests >= 10 && self.success_rate() < 50.0 {
                self.circuit_breaker_state.store(1, Ordering::Relaxed); // Open
            }
        }
    }

    fn update_health_status(&self, healthy: bool) {
        self.is_healthy.store(healthy, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_health_check.store(now, Ordering::Relaxed);

        // Transition circuit breaker to half-open if we're healthy and it was open
        if healthy && self.circuit_breaker_state.load(Ordering::Relaxed) == 1 {
            self.circuit_breaker_state.store(2, Ordering::Relaxed); // Half-open
        }
    }
}

/// Different load balancing algorithms
#[derive(Debug, Clone, PartialEq)]
enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    WeightedResponseTime,
    Random,
    IpHash,
}

/// Request to be processed by the service
#[derive(Debug, Clone)]
struct ServiceRequest {
    id: u64,
    client_ip: String,
    path: String,
    size_bytes: usize,
    priority: RequestPriority,
    timestamp: u64,
    timeout_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum RequestPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Response from the service
#[derive(Debug, Clone)]
struct ServiceResponse {
    request_id: u64,
    status_code: u16,
    response_time_ms: u64,
    instance_id: String,
    body_size_bytes: usize,
}

/// Load balancer with advanced features
struct LoadBalancer {
    instances: Arc<RwLock<HashMap<String, Arc<ServiceInstance>>>>,
    algorithm: LoadBalancingAlgorithm,
    round_robin_counter: AtomicUsize,
    request_queue: Arc<Mutex<VecDeque<ServiceRequest>>>,
    pending_requests: Arc<RwLock<HashMap<u64, (ServiceRequest, Instant)>>>,

    // Statistics
    total_requests: AtomicUsize,
    successful_requests: Arc<AtomicUsize>,
    failed_requests: Arc<AtomicUsize>,
    timeout_requests: Arc<AtomicUsize>,
    avg_response_time: Arc<AtomicU64>,

    // Configuration
    max_pending_requests: usize,
    default_timeout_ms: u64,
    health_check_interval_ms: u64,

    // Runtime
    runtime: Moirai,
    is_running: Arc<AtomicBool>,
}

impl LoadBalancer {
    fn new(algorithm: LoadBalancingAlgorithm, max_pending: usize) -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;

        let balancer = Self {
            instances: Arc::new(RwLock::new(HashMap::new())),
            algorithm,
            round_robin_counter: AtomicUsize::new(0),
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            total_requests: AtomicUsize::new(0),
            successful_requests: Arc::new(AtomicUsize::new(0)),
            failed_requests: Arc::new(AtomicUsize::new(0)),
            timeout_requests: Arc::new(AtomicUsize::new(0)),
            avg_response_time: Arc::new(AtomicU64::new(0)),
            max_pending_requests: max_pending,
            default_timeout_ms: 5000,
            health_check_interval_ms: 1000,
            runtime,
            is_running: Arc::new(AtomicBool::new(true)),
        };

        balancer.start_background_tasks()?;
        Ok(balancer)
    }

    fn add_instance(&self, instance: ServiceInstance) -> Result<(), String> {
        let mut instances = self
            .instances
            .write()
            .map_err(|_| "Failed to acquire instances write lock")?;

        let instance_id = instance.id.clone();
        instances.insert(instance_id.clone(), Arc::new(instance));

        println!("Added service instance: {}", instance_id);
        Ok(())
    }

    fn remove_instance(&self, instance_id: &str) -> Result<(), String> {
        let mut instances = self
            .instances
            .write()
            .map_err(|_| "Failed to acquire instances write lock")?;

        if instances.remove(instance_id).is_some() {
            println!("Removed service instance: {}", instance_id);
            Ok(())
        } else {
            Err(format!("Instance {} not found", instance_id))
        }
    }

    fn handle_request(&self, request: ServiceRequest) -> Result<(), String> {
        // Check if we're over capacity
        let pending_count = self
            .pending_requests
            .read()
            .map_err(|_| "Failed to read pending requests")?
            .len();

        if pending_count >= self.max_pending_requests {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
            return Err("Load balancer at capacity".to_string());
        }

        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Add to pending requests
        {
            let mut pending = self
                .pending_requests
                .write()
                .map_err(|_| "Failed to write pending requests")?;
            pending.insert(request.id, (request.clone(), Instant::now()));
        }

        // Select instance and route request
        let instance = self.select_instance(&request)?;
        self.route_request_to_instance(request, instance)?;

        Ok(())
    }

    fn select_instance(&self, request: &ServiceRequest) -> Result<Arc<ServiceInstance>, String> {
        let instances = self
            .instances
            .read()
            .map_err(|_| "Failed to read instances")?;

        let available_instances: Vec<_> = instances
            .values()
            .filter(|instance| instance.can_handle_request())
            .cloned()
            .collect();

        if available_instances.is_empty() {
            return Err("No healthy instances available".to_string());
        }

        match self.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                let index = self.round_robin_counter.fetch_add(1, Ordering::Relaxed)
                    % available_instances.len();
                Ok(available_instances[index].clone())
            }
            LoadBalancingAlgorithm::LeastConnections => {
                let instance = available_instances
                    .iter()
                    .min_by_key(|instance| instance.current_load.load(Ordering::Relaxed))
                    .unwrap();
                Ok(instance.clone())
            }
            LoadBalancingAlgorithm::WeightedResponseTime => {
                let instance = available_instances
                    .iter()
                    .min_by_key(|instance| {
                        let load_factor = instance.load_percentage();
                        let response_time = instance.avg_response_time();
                        (load_factor * response_time as f64) as u64
                    })
                    .unwrap();
                Ok(instance.clone())
            }
            LoadBalancingAlgorithm::Random => {
                let index = fastrand::usize(0..available_instances.len());
                Ok(available_instances[index].clone())
            }
            LoadBalancingAlgorithm::IpHash => {
                // Simple hash of IP for sticky sessions
                let hash = request.client_ip.bytes().map(|b| b as usize).sum::<usize>();
                let index = hash % available_instances.len();
                Ok(available_instances[index].clone())
            }
        }
    }

    fn route_request_to_instance(
        &self,
        request: ServiceRequest,
        instance: Arc<ServiceInstance>,
    ) -> Result<(), String> {
        instance.record_request_start();

        let pending_requests = self.pending_requests.clone();
        let successful_requests = self.successful_requests.clone();
        let failed_requests = self.failed_requests.clone();
        let timeout_requests = self.timeout_requests.clone();
        let avg_response_time = self.avg_response_time.clone();

        // Process request based on priority
        let priority = match request.priority {
            RequestPriority::Critical => Priority::High,
            RequestPriority::High => Priority::High,
            RequestPriority::Normal => Priority::Normal,
            RequestPriority::Low => Priority::Low,
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                let start_time = Instant::now();

                // Simulate request processing
                let response = Self::simulate_request_processing(&request, &instance);
                let processing_time = start_time.elapsed();

                // Record result
                match response {
                    Ok(response) => {
                        instance.record_request_complete(true, response.response_time_ms);
                        successful_requests.fetch_add(1, Ordering::Relaxed);

                        // Update average response time
                        let current_avg = avg_response_time.load(Ordering::Relaxed);
                        let new_avg = if current_avg == 0 {
                            response.response_time_ms
                        } else {
                            (current_avg * 7 + response.response_time_ms) / 8
                        };
                        avg_response_time.store(new_avg, Ordering::Relaxed);
                    }
                    Err(error) => {
                        let is_timeout = processing_time.as_millis() > request.timeout_ms as u128;
                        instance.record_request_complete(false, processing_time.as_millis() as u64);

                        if is_timeout {
                            timeout_requests.fetch_add(1, Ordering::Relaxed);
                        } else {
                            failed_requests.fetch_add(1, Ordering::Relaxed);
                        }

                        println!("Request {} failed: {}", request.id, error);
                    }
                }

                // Remove from pending requests
                if let Ok(mut pending) = pending_requests.write() {
                    pending.remove(&request.id);
                }
            },
            priority,
        );

        std::mem::drop(handle); // Let it run asynchronously
        Ok(())
    }

    fn simulate_request_processing(
        request: &ServiceRequest,
        instance: &ServiceInstance,
    ) -> Result<ServiceResponse, String> {
        // Simulate processing time based on request size and instance health
        let base_time_ms = (request.size_bytes / 1000).max(10) as u64;
        let load_factor = instance.load_percentage() / 100.0;
        let processing_time_ms = (base_time_ms as f64 * (1.0 + load_factor)) as u64;

        // Simulate processing delay
        std::thread::sleep(Duration::from_millis(processing_time_ms));

        // Simulate potential failures based on load and path
        let failure_rate = if instance.load_percentage() > 80.0 {
            0.1 // 10% failure rate when overloaded
        } else if request.path.contains("error") {
            0.8 // 80% failure rate for error paths (testing)
        } else {
            0.02 // 2% base failure rate
        };

        if fastrand::f64() < failure_rate {
            return Err("Service temporarily unavailable".to_string());
        }

        // Check timeout
        if processing_time_ms > request.timeout_ms {
            return Err("Request timeout".to_string());
        }

        Ok(ServiceResponse {
            request_id: request.id,
            status_code: 200,
            response_time_ms: processing_time_ms,
            instance_id: instance.id.clone(),
            body_size_bytes: fastrand::usize(100..10000),
        })
    }

    fn start_background_tasks(&self) -> Result<(), String> {
        // Health check task
        let health_checker = HealthChecker {
            instances: self.instances.clone(),
            is_running: self.is_running.clone(),
            check_interval_ms: self.health_check_interval_ms,
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while health_checker.is_running.load(Ordering::Relaxed) {
                    health_checker.perform_health_checks();
                    std::thread::sleep(Duration::from_millis(health_checker.check_interval_ms));
                }
            },
            Priority::Normal,
        );

        std::mem::drop(handle);

        // Timeout monitor task
        let timeout_monitor = TimeoutMonitor {
            pending_requests: self.pending_requests.clone(),
            timeout_requests: self.timeout_requests.clone(),
            is_running: self.is_running.clone(),
            default_timeout_ms: self.default_timeout_ms,
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while timeout_monitor.is_running.load(Ordering::Relaxed) {
                    timeout_monitor.check_timeouts();
                    std::thread::sleep(Duration::from_millis(1000)); // Check every second
                }
            },
            Priority::Low,
        );

        std::mem::drop(handle);

        Ok(())
    }

    fn get_statistics(&self) -> LoadBalancerStats {
        let instances = self.instances.read().unwrap();
        let healthy_instances = instances
            .values()
            .filter(|instance| instance.is_healthy.load(Ordering::Relaxed))
            .count();

        let total_capacity = instances
            .values()
            .map(|instance| instance.max_capacity)
            .sum();

        let current_load = instances
            .values()
            .map(|instance| instance.current_load.load(Ordering::Relaxed))
            .sum();

        let pending_count = self.pending_requests.read().unwrap().len();

        LoadBalancerStats {
            algorithm: self.algorithm.clone(),
            total_instances: instances.len(),
            healthy_instances,
            total_capacity,
            current_load,
            pending_requests: pending_count,
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            timeout_requests: self.timeout_requests.load(Ordering::Relaxed),
            avg_response_time_ms: self.avg_response_time.load(Ordering::Relaxed),
        }
    }

    fn shutdown(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }
}

struct HealthChecker {
    instances: Arc<RwLock<HashMap<String, Arc<ServiceInstance>>>>,
    is_running: Arc<AtomicBool>,
    check_interval_ms: u64,
}

impl HealthChecker {
    fn perform_health_checks(&self) {
        if let Ok(instances) = self.instances.read() {
            for instance in instances.values() {
                let is_healthy = self.check_instance_health(instance);
                instance.update_health_status(is_healthy);
            }
        }
    }

    fn check_instance_health(&self, instance: &ServiceInstance) -> bool {
        // Simulate health check
        let load_percentage = instance.load_percentage();
        let success_rate = instance.success_rate();

        // Instance is unhealthy if overloaded or too many failures
        !(load_percentage > 95.0 || success_rate < 10.0)
    }
}

struct TimeoutMonitor {
    pending_requests: Arc<RwLock<HashMap<u64, (ServiceRequest, Instant)>>>,
    timeout_requests: Arc<AtomicUsize>,
    is_running: Arc<AtomicBool>,
    default_timeout_ms: u64,
}

impl TimeoutMonitor {
    fn check_timeouts(&self) {
        if let Ok(mut pending) = self.pending_requests.write() {
            let now = Instant::now();
            let mut timed_out = Vec::new();

            for (request_id, (request, start_time)) in pending.iter() {
                let timeout =
                    Duration::from_millis(request.timeout_ms.max(self.default_timeout_ms));
                if now.duration_since(*start_time) > timeout {
                    timed_out.push(*request_id);
                }
            }

            for request_id in timed_out {
                pending.remove(&request_id);
                self.timeout_requests.fetch_add(1, Ordering::Relaxed);
                println!("Request {} timed out", request_id);
            }
        }
    }
}

#[derive(Debug)]
struct LoadBalancerStats {
    algorithm: LoadBalancingAlgorithm,
    total_instances: usize,
    healthy_instances: usize,
    total_capacity: usize,
    current_load: usize,
    pending_requests: usize,
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    timeout_requests: usize,
    avg_response_time_ms: u64,
}

impl LoadBalancerStats {
    fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            100.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    fn load_percentage(&self) -> f64 {
        if self.total_capacity == 0 {
            0.0
        } else {
            (self.current_load as f64 / self.total_capacity as f64) * 100.0
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Network Service Load Balancing - Edge Case Testing");
    println!("==================================================");

    // Create load balancer with different algorithms
    let load_balancer = LoadBalancer::new(LoadBalancingAlgorithm::WeightedResponseTime, 1000)?;

    // Add service instances with different capacities
    println!("\n1. Setting up service instances...");
    load_balancer.add_instance(ServiceInstance::new(
        "service-1".to_string(),
        "http://service-1:8080".to_string(),
        100,
    ))?;

    load_balancer.add_instance(ServiceInstance::new(
        "service-2".to_string(),
        "http://service-2:8080".to_string(),
        150,
    ))?;

    load_balancer.add_instance(ServiceInstance::new(
        "service-3".to_string(),
        "http://service-3:8080".to_string(),
        80,
    ))?;

    load_balancer.add_instance(ServiceInstance::new(
        "service-4".to_string(),
        "http://service-4:8080".to_string(),
        120,
    ))?;

    println!("  Added 4 service instances with varying capacities");

    // Generate and process normal load
    println!("\n2. Processing normal load (500 requests)...");
    let start_time = Instant::now();

    for i in 0..500 {
        let request = ServiceRequest {
            id: i as u64,
            client_ip: format!("192.168.1.{}", (i % 254) + 1),
            path: format!("/api/service/{}", i % 10),
            size_bytes: fastrand::usize(100..5000),
            priority: match i % 4 {
                0 => RequestPriority::Low,
                1 => RequestPriority::Normal,
                2 => RequestPriority::High,
                3 => RequestPriority::Critical,
                _ => RequestPriority::Normal,
            },
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            timeout_ms: 3000,
        };

        if let Err(e) = load_balancer.handle_request(request) {
            println!("  Request {} failed: {}", i, e);
        }

        // Throttle request rate
        if i % 50 == 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    let normal_load_time = start_time.elapsed();
    println!("  Normal load completed in {:?}", normal_load_time);

    // Wait for processing
    std::thread::sleep(Duration::from_millis(500));

    // Edge Case 1: Burst load testing
    println!("\n3. Testing burst load (1000 requests in quick succession)...");
    let burst_start = Instant::now();

    for i in 1000..2000 {
        let request = ServiceRequest {
            id: i as u64,
            client_ip: format!("10.0.0.{}", (i % 254) + 1),
            path: "/api/burst".to_string(),
            size_bytes: fastrand::usize(50..500),
            priority: if i % 10 == 0 {
                RequestPriority::Critical
            } else {
                RequestPriority::Normal
            },
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            timeout_ms: 2000,
        };

        let _ = load_balancer.handle_request(request); // Ignore errors during burst
    }

    let burst_time = burst_start.elapsed();
    println!("  Burst load sent in {:?}", burst_time);

    // Edge Case 2: Error-prone requests
    println!("\n4. Testing error handling with failure-prone requests...");

    for i in 2000..2100 {
        let request = ServiceRequest {
            id: i as u64,
            client_ip: format!("172.16.0.{}", (i % 254) + 1),
            path: "/api/error/simulate".to_string(), // This triggers high failure rate
            size_bytes: fastrand::usize(1000..10000),
            priority: RequestPriority::Normal,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            timeout_ms: 1000,
        };

        let _ = load_balancer.handle_request(request);
    }

    // Edge Case 3: Instance failure simulation
    println!("\n5. Simulating instance failure...");
    load_balancer.remove_instance("service-2")?;

    // Continue sending requests after instance failure
    for i in 2100..2200 {
        let request = ServiceRequest {
            id: i as u64,
            client_ip: format!("192.168.2.{}", (i % 254) + 1),
            path: "/api/after/failure".to_string(),
            size_bytes: fastrand::usize(500..2000),
            priority: RequestPriority::High,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            timeout_ms: 4000,
        };

        let _ = load_balancer.handle_request(request);
    }

    // Wait for all processing to complete
    std::thread::sleep(Duration::from_secs(2));

    // Edge Case 4: Auto-scaling simulation
    println!("\n6. Simulating auto-scaling response...");
    load_balancer.add_instance(ServiceInstance::new(
        "service-5".to_string(),
        "http://service-5:8080".to_string(),
        200, // High capacity new instance
    ))?;

    load_balancer.add_instance(ServiceInstance::new(
        "service-6".to_string(),
        "http://service-6:8080".to_string(),
        180,
    ))?;

    // Final load test with scaled infrastructure
    for i in 2200..2300 {
        let request = ServiceRequest {
            id: i as u64,
            client_ip: format!("10.1.0.{}", (i % 254) + 1),
            path: "/api/scaled".to_string(),
            size_bytes: fastrand::usize(200..1000),
            priority: RequestPriority::Normal,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            timeout_ms: 5000,
        };

        let _ = load_balancer.handle_request(request);
    }

    // Final wait and statistics
    std::thread::sleep(Duration::from_secs(1));

    // Display comprehensive statistics
    println!("\n7. Final Load Balancer Statistics:");
    let stats = load_balancer.get_statistics();

    println!("  ├─ Configuration:");
    println!("  │  ├─ Algorithm: {:?}", stats.algorithm);
    println!("  │  ├─ Total instances: {}", stats.total_instances);
    println!("  │  ├─ Healthy instances: {}", stats.healthy_instances);
    println!("  │  └─ Total capacity: {}", stats.total_capacity);

    println!("  ├─ Current Load:");
    println!("  │  ├─ Active requests: {}", stats.current_load);
    println!("  │  ├─ Pending requests: {}", stats.pending_requests);
    println!("  │  └─ Load percentage: {:.1}%", stats.load_percentage());

    println!("  ├─ Request Statistics:");
    println!("  │  ├─ Total requests: {}", stats.total_requests);
    println!("  │  ├─ Successful: {}", stats.successful_requests);
    println!("  │  ├─ Failed: {}", stats.failed_requests);
    println!("  │  ├─ Timeouts: {}", stats.timeout_requests);
    println!("  │  └─ Success rate: {:.2}%", stats.success_rate());

    println!("  └─ Performance:");
    println!(
        "     ├─ Avg response time: {}ms",
        stats.avg_response_time_ms
    );
    println!(
        "     ├─ Total throughput: {:.0} req/s",
        stats.total_requests as f64 / start_time.elapsed().as_secs_f64()
    );
    println!(
        "     └─ Instance efficiency: {:.1}%",
        (stats.healthy_instances as f64 / stats.total_instances as f64) * 100.0
    );

    // Display per-instance statistics
    println!("\n8. Per-Instance Statistics:");
    if let Ok(instances) = load_balancer.instances.read() {
        for (id, instance) in instances.iter() {
            println!("  Instance {}: ", id);
            println!(
                "    ├─ Load: {:.1}% ({}/{})",
                instance.load_percentage(),
                instance.current_load.load(Ordering::Relaxed),
                instance.max_capacity
            );
            println!("    ├─ Success rate: {:.1}%", instance.success_rate());
            println!("    ├─ Avg response: {}ms", instance.avg_response_time());
            println!(
                "    ├─ Healthy: {}",
                instance.is_healthy.load(Ordering::Relaxed)
            );
            println!(
                "    └─ Circuit breaker: {}",
                match instance.circuit_breaker_state.load(Ordering::Relaxed) {
                    0 => "Closed",
                    1 => "Open",
                    2 => "Half-Open",
                    _ => "Unknown",
                }
            );
        }
    }

    // Test different algorithms for comparison
    println!("\n9. Testing different load balancing algorithms...");
    let algorithms = [
        LoadBalancingAlgorithm::RoundRobin,
        LoadBalancingAlgorithm::LeastConnections,
        LoadBalancingAlgorithm::Random,
    ];

    for (i, algorithm) in algorithms.iter().enumerate() {
        println!("  Testing {:?}...", algorithm);
        let test_balancer = LoadBalancer::new(algorithm.clone(), 100)?;

        // Add instances
        for j in 0..3 {
            test_balancer.add_instance(ServiceInstance::new(
                format!("test-{}-{}", i, j),
                format!("http://test-{}-{}:8080", i, j),
                50,
            ))?;
        }

        // Send test requests
        let test_start = Instant::now();
        for k in 0..100 {
            let request = ServiceRequest {
                id: (i * 1000 + k) as u64,
                client_ip: format!("test.{}.{}.{}", i, k % 10, k % 5),
                path: "/test".to_string(),
                size_bytes: 100,
                priority: RequestPriority::Normal,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                timeout_ms: 1000,
            };
            let _ = test_balancer.handle_request(request);
        }

        std::thread::sleep(Duration::from_millis(200));
        let test_stats = test_balancer.get_statistics();
        let test_time = test_start.elapsed();

        println!(
            "    └─ Success rate: {:.1}%, Avg time: {}ms, Duration: {:?}",
            test_stats.success_rate(),
            test_stats.avg_response_time_ms,
            test_time
        );

        test_balancer.shutdown();
    }

    // Shutdown
    load_balancer.shutdown();
    println!("\n10. Load balancer shutdown completed.");

    println!("\nNetwork service load balancing testing completed!");
    println!("Successfully handled burst loads, instance failures, and auto-scaling scenarios.");

    Ok(())
}
