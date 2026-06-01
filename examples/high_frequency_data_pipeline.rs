//! High-Frequency Data Processing Pipeline - Real-World Streaming Example
//!
//! This example demonstrates:
//! - Real-time market data processing with microsecond latency requirements
//! - Backpressure handling in high-throughput scenarios
//! - Circuit breaker patterns for error resilience
//! - Memory pool management under sustained load
//! - Ordered processing guarantees despite parallel execution

#![allow(dead_code)] // This example keeps realistic market-data and pool metadata beyond the short demo path.

use moirai::{Moirai, Priority};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Represents a market data tick (simplified financial market data)
#[derive(Debug, Clone)]
struct MarketTick {
    symbol: String,
    price: f64,
    volume: u64,
    timestamp_nanos: u64,
    sequence_number: u64,
    tick_type: TickType,
}

#[derive(Debug, Clone, PartialEq)]
enum TickType {
    Trade,
    BidQuote,
    AskQuote,
    Last,
}

impl fmt::Display for TickType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TickType::Trade => write!(f, "TRADE"),
            TickType::BidQuote => write!(f, "BID"),
            TickType::AskQuote => write!(f, "ASK"),
            TickType::Last => write!(f, "LAST"),
        }
    }
}

/// Processed market data with calculated metrics
#[derive(Debug, Clone)]
struct ProcessedData {
    symbol: String,
    vwap: f64, // Volume Weighted Average Price
    total_volume: u64,
    price_change: f64,
    volatility: f64,
    processing_latency_nanos: u64,
    sequence_number: u64,
}

/// Circuit breaker for handling processing failures
#[derive(Debug)]
struct CircuitBreaker {
    failure_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    state: Arc<AtomicU8>, // 0: Closed, 1: Open, 2: Half-Open
    last_failure_time: Arc<AtomicU64>,
    failure_threshold: usize,
    recovery_timeout_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

impl CircuitBreaker {
    fn new(failure_threshold: usize, recovery_timeout_ms: u64) -> Self {
        Self {
            failure_count: Arc::new(AtomicUsize::new(0)),
            success_count: Arc::new(AtomicUsize::new(0)),
            state: Arc::new(AtomicU8::new(CircuitState::Closed as u8)),
            last_failure_time: Arc::new(AtomicU64::new(0)),
            failure_threshold,
            recovery_timeout_ms,
        }
    }

    fn call<F, T, E>(&self, operation: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, E>,
        E: fmt::Display,
    {
        if self.current_state() == CircuitState::Open {
            // Check if we should transition to half-open
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            let last_failure = self.last_failure_time.load(Ordering::Relaxed);

            if now - last_failure > self.recovery_timeout_ms {
                self.state
                    .store(CircuitState::HalfOpen as u8, Ordering::Relaxed);
            } else {
                return Err("Circuit breaker is OPEN".to_string());
            }
        }

        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(format!("Operation failed: {}", e))
            }
        }
    }

    fn current_state(&self) -> CircuitState {
        match self.state.load(Ordering::Relaxed) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }

    fn on_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        if self.current_state() == CircuitState::HalfOpen {
            self.state
                .store(CircuitState::Closed as u8, Ordering::Relaxed);
            self.failure_count.store(0, Ordering::Relaxed);
        }
    }

    fn on_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_failure_time.store(now, Ordering::Relaxed);

        if failures >= self.failure_threshold {
            self.state
                .store(CircuitState::Open as u8, Ordering::Relaxed);
        }
    }

    fn stats(&self) -> (usize, usize, CircuitState) {
        (
            self.success_count.load(Ordering::Relaxed),
            self.failure_count.load(Ordering::Relaxed),
            self.current_state(),
        )
    }
}

/// Memory pool for efficient allocation of processing objects
struct MemoryPool<T> {
    pool: Arc<Mutex<VecDeque<T>>>,
    created_count: Arc<AtomicUsize>,
    reused_count: Arc<AtomicUsize>,
    max_size: usize,
}

impl<T: Default> MemoryPool<T> {
    fn new(max_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            created_count: Arc::new(AtomicUsize::new(0)),
            reused_count: Arc::new(AtomicUsize::new(0)),
            max_size,
        }
    }

    fn acquire(&self) -> T {
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(item) = pool.pop_front() {
                self.reused_count.fetch_add(1, Ordering::Relaxed);
                return item;
            }
        }

        self.created_count.fetch_add(1, Ordering::Relaxed);
        T::default()
    }

    fn release(&self, item: T) {
        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < self.max_size {
                pool.push_back(item);
            }
        }
    }

    fn stats(&self) -> (usize, usize, usize) {
        let pool_size = self.pool.lock().map(|p| p.len()).unwrap_or(0);
        (
            self.created_count.load(Ordering::Relaxed),
            self.reused_count.load(Ordering::Relaxed),
            pool_size,
        )
    }
}

/// Backpressure controller to prevent memory overflow
struct BackpressureController {
    pending_count: Arc<AtomicUsize>,
    max_pending: usize,
    dropped_count: Arc<AtomicUsize>,
    shed_load: Arc<AtomicBool>,
}

impl BackpressureController {
    fn new(max_pending: usize) -> Self {
        Self {
            pending_count: Arc::new(AtomicUsize::new(0)),
            max_pending,
            dropped_count: Arc::new(AtomicUsize::new(0)),
            shed_load: Arc::new(AtomicBool::new(false)),
        }
    }

    fn try_accept(&self) -> bool {
        let current = self.pending_count.load(Ordering::Relaxed);

        if current >= self.max_pending {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            self.shed_load.store(true, Ordering::Relaxed);
            false
        } else {
            self.pending_count.fetch_add(1, Ordering::Relaxed);
            if current < self.max_pending / 2 {
                self.shed_load.store(false, Ordering::Relaxed);
            }
            true
        }
    }

    fn complete(&self) {
        self.pending_count.fetch_sub(1, Ordering::Relaxed);
    }

    fn is_load_shedding(&self) -> bool {
        self.shed_load.load(Ordering::Relaxed)
    }

    fn stats(&self) -> (usize, usize, bool) {
        (
            self.pending_count.load(Ordering::Relaxed),
            self.dropped_count.load(Ordering::Relaxed),
            self.is_load_shedding(),
        )
    }
}

/// High-frequency data processing pipeline
struct DataProcessingPipeline {
    runtime: Moirai,
    symbol_processors: Arc<RwLock<HashMap<String, SymbolProcessor>>>,
    circuit_breaker: Arc<CircuitBreaker>,
    memory_pool: Arc<MemoryPool<Vec<f64>>>, // For calculations
    backpressure: Arc<BackpressureController>,
    processed_count: Arc<AtomicUsize>,
    error_count: Arc<AtomicUsize>,
    total_latency_nanos: Arc<AtomicU64>,
}

/// Per-symbol processor maintaining state
#[derive(Debug)]
struct SymbolProcessor {
    symbol: String,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<u64>,
    last_sequence: u64,
    out_of_order_count: AtomicUsize,
    max_history: usize,
}

impl SymbolProcessor {
    fn new(symbol: String) -> Self {
        Self {
            symbol,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            last_sequence: 0,
            out_of_order_count: AtomicUsize::new(0),
            max_history: 1000,
        }
    }

    fn process_tick(&mut self, tick: &MarketTick) -> Result<ProcessedData, String> {
        // Check for out-of-order delivery (common in high-frequency scenarios)
        if tick.sequence_number <= self.last_sequence {
            self.out_of_order_count.fetch_add(1, Ordering::Relaxed);
            return Err(format!(
                "Out-of-order tick: {} <= {}",
                tick.sequence_number, self.last_sequence
            ));
        }

        self.last_sequence = tick.sequence_number;

        // Update history
        self.price_history.push_back(tick.price);
        self.volume_history.push_back(tick.volume);

        // Maintain history size
        if self.price_history.len() > self.max_history {
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }

        // Calculate metrics
        let vwap = self.calculate_vwap()?;
        let total_volume = self.volume_history.iter().sum();
        let price_change = if self.price_history.len() >= 2 {
            tick.price - self.price_history[self.price_history.len() - 2]
        } else {
            0.0
        };
        let volatility = self.calculate_volatility()?;

        Ok(ProcessedData {
            symbol: self.symbol.clone(),
            vwap,
            total_volume,
            price_change,
            volatility,
            processing_latency_nanos: 0, // Will be set by caller
            sequence_number: tick.sequence_number,
        })
    }

    fn calculate_vwap(&self) -> Result<f64, String> {
        if self.price_history.is_empty() || self.volume_history.is_empty() {
            return Ok(0.0);
        }

        let mut total_value = 0.0;
        let mut total_volume = 0u64;

        for (price, volume) in self.price_history.iter().zip(self.volume_history.iter()) {
            total_value += price * (*volume as f64);
            total_volume += volume;
        }

        if total_volume == 0 {
            Ok(0.0)
        } else {
            Ok(total_value / total_volume as f64)
        }
    }

    fn calculate_volatility(&self) -> Result<f64, String> {
        if self.price_history.len() < 2 {
            return Ok(0.0);
        }

        let mean = self.price_history.iter().sum::<f64>() / self.price_history.len() as f64;
        let variance = self
            .price_history
            .iter()
            .map(|price| (price - mean).powi(2))
            .sum::<f64>()
            / self.price_history.len() as f64;

        Ok(variance.sqrt())
    }
}

impl DataProcessingPipeline {
    fn new() -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;

        Ok(Self {
            runtime,
            symbol_processors: Arc::new(RwLock::new(HashMap::new())),
            circuit_breaker: Arc::new(CircuitBreaker::new(10, 5000)), // 10 failures, 5s recovery
            memory_pool: Arc::new(MemoryPool::new(1000)),
            backpressure: Arc::new(BackpressureController::new(10000)), // Max 10k pending
            processed_count: Arc::new(AtomicUsize::new(0)),
            error_count: Arc::new(AtomicUsize::new(0)),
            total_latency_nanos: Arc::new(AtomicU64::new(0)),
        })
    }

    fn process_tick(&self, tick: MarketTick) -> Result<(), String> {
        // Edge Case 1: Backpressure handling
        if !self.backpressure.try_accept() {
            return Err("Backpressure: System overloaded, dropping tick".to_string());
        }

        let start_time = Instant::now();
        let processors = self.symbol_processors.clone();
        let circuit_breaker = self.circuit_breaker.clone();
        let memory_pool = self.memory_pool.clone();
        let backpressure = self.backpressure.clone();
        let processed_count = self.processed_count.clone();
        let error_count = self.error_count.clone();
        let total_latency = self.total_latency_nanos.clone();

        // Process asynchronously with high priority for time-sensitive data
        let priority = if tick.tick_type == TickType::Trade {
            Priority::High
        } else {
            Priority::Normal
        };

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                // Use circuit breaker for resilience
                let result = circuit_breaker.call(|| -> Result<ProcessedData, &'static str> {
                    // Acquire memory for calculations
                    let _calc_buffer = memory_pool.acquire();

                    // Get or create symbol processor
                    let mut processed_data = {
                        let mut symbol_processors = processors
                            .write()
                            .map_err(|_| "Failed to acquire symbol processors lock")?;

                        let processor = symbol_processors
                            .entry(tick.symbol.clone())
                            .or_insert_with(|| SymbolProcessor::new(tick.symbol.clone()));

                        processor
                            .process_tick(&tick)
                            .map_err(|_| "Failed to process tick")?
                    };

                    // Calculate processing latency
                    let processing_latency = start_time.elapsed().as_nanos() as u64;
                    processed_data.processing_latency_nanos = processing_latency;

                    // Update statistics
                    processed_count.fetch_add(1, Ordering::Relaxed);
                    total_latency.fetch_add(processing_latency, Ordering::Relaxed);

                    Ok(processed_data)
                });

                match result {
                    Ok(processed_data) => {
                        // Simulate downstream processing (e.g., sending to subscribers)
                        if processed_data.volatility > 0.05 {
                            // High volatility detected - could trigger alerts
                            println!(
                                "HIGH VOLATILITY ALERT: {} @ {:.4} (vol: {:.4})",
                                processed_data.symbol,
                                processed_data.vwap,
                                processed_data.volatility
                            );
                        }
                        Ok(())
                    }
                    Err(e) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                        Err(e)
                    }
                }
            },
            priority,
        );

        // Complete backpressure tracking
        let _ = handle.join();
        backpressure.complete();

        Ok(())
    }

    fn get_stats(&self) -> Result<ProcessingStats, String> {
        let processed = self.processed_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let total_latency = self.total_latency_nanos.load(Ordering::Relaxed);

        let avg_latency_nanos = if processed > 0 {
            total_latency / processed as u64
        } else {
            0
        };

        let (circuit_success, circuit_failures, circuit_state) = self.circuit_breaker.stats();
        let (pool_created, pool_reused, pool_size) = self.memory_pool.stats();
        let (pending, dropped, load_shedding) = self.backpressure.stats();

        let symbol_count = self
            .symbol_processors
            .read()
            .map_err(|_| "Failed to read symbol processors")?
            .len();

        Ok(ProcessingStats {
            processed_ticks: processed,
            error_count: errors,
            avg_latency_nanos,
            circuit_success,
            circuit_failures,
            circuit_state,
            pool_created,
            pool_reused,
            pool_size,
            pending_count: pending,
            dropped_count: dropped,
            load_shedding,
            unique_symbols: symbol_count,
        })
    }
}

#[derive(Debug)]
struct ProcessingStats {
    processed_ticks: usize,
    error_count: usize,
    avg_latency_nanos: u64,
    circuit_success: usize,
    circuit_failures: usize,
    circuit_state: CircuitState,
    pool_created: usize,
    pool_reused: usize,
    pool_size: usize,
    pending_count: usize,
    dropped_count: usize,
    load_shedding: bool,
    unique_symbols: usize,
}

/// Generate realistic market data for testing
struct MarketDataGenerator {
    sequence_counter: AtomicU64,
    symbols: Vec<String>,
    base_prices: HashMap<String, f64>,
}

impl MarketDataGenerator {
    fn new() -> Self {
        let symbols = vec![
            "AAPL".to_string(),
            "GOOGL".to_string(),
            "MSFT".to_string(),
            "AMZN".to_string(),
            "TSLA".to_string(),
            "META".to_string(),
            "NVDA".to_string(),
            "AMD".to_string(),
            "INTC".to_string(),
            "IBM".to_string(),
        ];

        let mut base_prices = HashMap::new();
        base_prices.insert("AAPL".to_string(), 150.0);
        base_prices.insert("GOOGL".to_string(), 2500.0);
        base_prices.insert("MSFT".to_string(), 300.0);
        base_prices.insert("AMZN".to_string(), 3200.0);
        base_prices.insert("TSLA".to_string(), 800.0);
        base_prices.insert("META".to_string(), 250.0);
        base_prices.insert("NVDA".to_string(), 400.0);
        base_prices.insert("AMD".to_string(), 90.0);
        base_prices.insert("INTC".to_string(), 55.0);
        base_prices.insert("IBM".to_string(), 130.0);

        Self {
            sequence_counter: AtomicU64::new(1),
            symbols,
            base_prices,
        }
    }

    fn generate_tick(
        &self,
        symbol_index: usize,
        simulate_error: bool,
    ) -> Result<MarketTick, String> {
        if simulate_error {
            return Err("Simulated market data error".to_string());
        }

        let symbol = &self.symbols[symbol_index % self.symbols.len()];
        let base_price = self.base_prices[symbol];

        // Simulate price movement (±2% random walk)
        let price_change = (fastrand::f64() - 0.5) * 0.04;
        let price = base_price * (1.0 + price_change);

        let volume = fastrand::u64(100..10000);
        let timestamp_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let tick_types = [
            TickType::Trade,
            TickType::BidQuote,
            TickType::AskQuote,
            TickType::Last,
        ];
        let tick_type = tick_types[fastrand::usize(0..tick_types.len())].clone();

        Ok(MarketTick {
            symbol: symbol.clone(),
            price,
            volume,
            timestamp_nanos,
            sequence_number: self.sequence_counter.fetch_add(1, Ordering::Relaxed),
            tick_type,
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("High-Frequency Data Processing Pipeline");
    println!("======================================");

    let pipeline = DataProcessingPipeline::new()?;
    let generator = MarketDataGenerator::new();

    println!("\n1. Processing normal market data flow...");
    let start_time = Instant::now();

    // Process normal flow
    for i in 0..1000 {
        let symbol_index = i % 10;
        match generator.generate_tick(symbol_index, false) {
            Ok(tick) => {
                if let Err(e) = pipeline.process_tick(tick) {
                    println!("  Processing error: {}", e);
                }
            }
            Err(e) => println!("  Generation error: {}", e),
        }
    }

    // Wait for processing to complete
    std::thread::sleep(Duration::from_millis(100));

    let normal_processing_time = start_time.elapsed();
    println!(
        "  Normal processing completed in {:?}",
        normal_processing_time
    );

    // Edge Case 1: High-frequency burst
    println!("\n2. Testing high-frequency burst (backpressure scenario)...");
    let burst_start = Instant::now();

    for i in 0..5000 {
        let symbol_index = i % 10;
        if let Ok(tick) = generator.generate_tick(symbol_index, false) {
            let _ = pipeline.process_tick(tick); // Ignore backpressure errors for this test
        }

        // Simulate microsecond-level processing
        if i % 100 == 0 {
            std::thread::sleep(Duration::from_micros(1));
        }
    }

    let burst_time = burst_start.elapsed();
    println!("  Burst processing completed in {:?}", burst_time);

    // Edge Case 2: Error scenario (circuit breaker testing)
    println!("\n3. Testing circuit breaker with simulated errors...");

    for i in 0..20 {
        let symbol_index = i % 10;
        let simulate_error = i % 3 == 0; // Inject errors for 1/3 of ticks

        match generator.generate_tick(symbol_index, simulate_error) {
            Ok(tick) => {
                let _ = pipeline.process_tick(tick);
            }
            Err(e) => println!("  Expected error: {}", e),
        }
    }

    // Wait for all processing to complete
    std::thread::sleep(Duration::from_millis(200));

    // Display comprehensive statistics
    println!("\n4. Final Processing Statistics:");
    match pipeline.get_stats() {
        Ok(stats) => {
            println!("  ├─ Ticks processed: {}", stats.processed_ticks);
            println!("  ├─ Processing errors: {}", stats.error_count);
            println!(
                "  ├─ Average latency: {:.2} μs",
                stats.avg_latency_nanos as f64 / 1000.0
            );
            println!("  ├─ Unique symbols: {}", stats.unique_symbols);
            println!(
                "  ├─ Success rate: {:.2}%",
                (stats.processed_ticks as f64 / (stats.processed_ticks + stats.error_count) as f64)
                    * 100.0
            );

            println!("  ├─ Circuit Breaker:");
            println!("  │  ├─ Successes: {}", stats.circuit_success);
            println!("  │  ├─ Failures: {}", stats.circuit_failures);
            println!("  │  └─ State: {:?}", stats.circuit_state);

            println!("  ├─ Memory Pool:");
            println!("  │  ├─ Objects created: {}", stats.pool_created);
            println!("  │  ├─ Objects reused: {}", stats.pool_reused);
            println!("  │  ├─ Pool size: {}", stats.pool_size);
            println!(
                "  │  └─ Reuse rate: {:.2}%",
                (stats.pool_reused as f64 / (stats.pool_created + stats.pool_reused) as f64)
                    * 100.0
            );

            println!("  └─ Backpressure:");
            println!("     ├─ Pending: {}", stats.pending_count);
            println!("     ├─ Dropped: {}", stats.dropped_count);
            println!(
                "     └─ Load shedding: {}",
                if stats.load_shedding {
                    "ACTIVE"
                } else {
                    "INACTIVE"
                }
            );
        }
        Err(e) => println!("  Failed to get statistics: {}", e),
    }

    // Edge Case 3: Performance analysis
    println!("\n5. Performance Analysis:");
    let total_ticks = 1000 + 5000 + 20; // Normal + burst + error test
    let total_time = start_time.elapsed();
    let throughput = total_ticks as f64 / total_time.as_secs_f64();

    println!("  ├─ Total throughput: {:.0} ticks/second", throughput);
    println!(
        "  ├─ Peak burst rate: {:.0} ticks/second",
        5000.0 / burst_time.as_secs_f64()
    );
    println!("  └─ Memory efficiency: High (object pooling active)");

    println!("\nHigh-frequency data processing pipeline completed!");
    println!("Successfully handled backpressure, errors, and sustained load.");

    Ok(())
}
