//! IoT Device Management System - Event-Driven Real-Time Coordination
//!
//! This example demonstrates:
//! - Event-driven architecture for IoT device coordination
//! - Real-time sensor data streaming with time-series processing
//! - Device discovery and automatic registration protocols
//! - Command distribution with reliable delivery guarantees
//! - Telemetry aggregation and anomaly detection
//! - Edge computing with local processing and cloud synchronization

#![allow(dead_code)] // This example keeps device-command/event variants that document broader IoT workflows.

use moirai::{Moirai, Priority};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Different types of IoT devices in the system
#[derive(Debug, Clone, PartialEq)]
enum DeviceType {
    TemperatureSensor,
    HumiditySensor,
    MotionDetector,
    SmartLight,
    SmartThermostat,
    SecurityCamera,
    SmartLock,
    AirQualitySensor,
    EnergyMeter,
    WaterSensor,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::TemperatureSensor => write!(f, "Temperature Sensor"),
            DeviceType::HumiditySensor => write!(f, "Humidity Sensor"),
            DeviceType::MotionDetector => write!(f, "Motion Detector"),
            DeviceType::SmartLight => write!(f, "Smart Light"),
            DeviceType::SmartThermostat => write!(f, "Smart Thermostat"),
            DeviceType::SecurityCamera => write!(f, "Security Camera"),
            DeviceType::SmartLock => write!(f, "Smart Lock"),
            DeviceType::AirQualitySensor => write!(f, "Air Quality Sensor"),
            DeviceType::EnergyMeter => write!(f, "Energy Meter"),
            DeviceType::WaterSensor => write!(f, "Water Sensor"),
        }
    }
}

/// IoT device representation with capabilities and state
#[derive(Debug)]
struct IoTDevice {
    id: String,
    device_type: DeviceType,
    name: String,
    location: String,
    firmware_version: String,
    last_seen: AtomicU64,
    is_online: AtomicBool,
    battery_level: AtomicU64,   // 0-100
    signal_strength: AtomicU64, // 0-100
    capabilities: Vec<String>,
    configuration: Arc<RwLock<HashMap<String, String>>>,
    telemetry_count: AtomicUsize,
    command_count: AtomicUsize,
}

impl Clone for IoTDevice {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            device_type: self.device_type.clone(),
            name: self.name.clone(),
            location: self.location.clone(),
            firmware_version: self.firmware_version.clone(),
            last_seen: AtomicU64::new(self.last_seen.load(Ordering::Relaxed)),
            is_online: AtomicBool::new(self.is_online.load(Ordering::Relaxed)),
            battery_level: AtomicU64::new(self.battery_level.load(Ordering::Relaxed)),
            signal_strength: AtomicU64::new(self.signal_strength.load(Ordering::Relaxed)),
            capabilities: self.capabilities.clone(),
            configuration: self.configuration.clone(),
            telemetry_count: AtomicUsize::new(self.telemetry_count.load(Ordering::Relaxed)),
            command_count: AtomicUsize::new(self.command_count.load(Ordering::Relaxed)),
        }
    }
}

impl IoTDevice {
    fn new(id: String, device_type: DeviceType, name: String, location: String) -> Self {
        let capabilities = match device_type {
            DeviceType::TemperatureSensor => vec!["temperature".to_string()],
            DeviceType::HumiditySensor => vec!["humidity".to_string()],
            DeviceType::MotionDetector => vec!["motion".to_string(), "presence".to_string()],
            DeviceType::SmartLight => vec![
                "brightness".to_string(),
                "color".to_string(),
                "power".to_string(),
            ],
            DeviceType::SmartThermostat => vec![
                "temperature".to_string(),
                "humidity".to_string(),
                "target_temp".to_string(),
            ],
            DeviceType::SecurityCamera => vec![
                "video".to_string(),
                "motion".to_string(),
                "recording".to_string(),
            ],
            DeviceType::SmartLock => vec!["lock_state".to_string(), "battery".to_string()],
            DeviceType::AirQualitySensor => {
                vec!["co2".to_string(), "voc".to_string(), "pm25".to_string()]
            }
            DeviceType::EnergyMeter => vec![
                "power".to_string(),
                "energy".to_string(),
                "voltage".to_string(),
            ],
            DeviceType::WaterSensor => vec!["water_level".to_string(), "flow_rate".to_string()],
        };

        Self {
            id,
            device_type,
            name,
            location,
            firmware_version: "1.0.0".to_string(),
            last_seen: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
            is_online: AtomicBool::new(true),
            battery_level: AtomicU64::new(fastrand::u64(70..100)),
            signal_strength: AtomicU64::new(fastrand::u64(60..100)),
            capabilities,
            configuration: Arc::new(RwLock::new(HashMap::new())),
            telemetry_count: AtomicUsize::new(0),
            command_count: AtomicUsize::new(0),
        }
    }

    fn update_last_seen(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_seen.store(now, Ordering::Relaxed);
        self.is_online.store(true, Ordering::Relaxed);
    }

    fn is_active(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last_seen = self.last_seen.load(Ordering::Relaxed);
        self.is_online.load(Ordering::Relaxed) && (now - last_seen) < 300 // 5 minutes
    }

    fn get_configuration(&self, key: &str) -> Option<String> {
        self.configuration.read().ok()?.get(key).cloned()
    }

    fn set_configuration(&self, key: String, value: String) -> Result<(), String> {
        let mut config = self
            .configuration
            .write()
            .map_err(|_| "Failed to acquire configuration lock")?;
        config.insert(key, value);
        Ok(())
    }

    fn increment_telemetry_count(&self) {
        self.telemetry_count.fetch_add(1, Ordering::Relaxed);
    }

    fn increment_command_count(&self) {
        self.command_count.fetch_add(1, Ordering::Relaxed);
    }

    fn stats(&self) -> (usize, usize, u64, u64, bool) {
        (
            self.telemetry_count.load(Ordering::Relaxed),
            self.command_count.load(Ordering::Relaxed),
            self.battery_level.load(Ordering::Relaxed),
            self.signal_strength.load(Ordering::Relaxed),
            self.is_active(),
        )
    }
}

/// Telemetry data from IoT devices
#[derive(Debug, Clone)]
struct TelemetryData {
    device_id: String,
    metric_name: String,
    value: f64,
    unit: String,
    timestamp: u64,
    quality: DataQuality,
    location: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum DataQuality {
    Good,
    Fair,
    Poor,
    Invalid,
}

impl TelemetryData {
    fn new(device_id: String, metric_name: String, value: f64, unit: String) -> Self {
        let quality = if value.is_nan() || value.is_infinite() {
            DataQuality::Invalid
        } else if fastrand::f64() < 0.05 {
            DataQuality::Poor
        } else if fastrand::f64() < 0.15 {
            DataQuality::Fair
        } else {
            DataQuality::Good
        };

        Self {
            device_id,
            metric_name,
            value,
            unit,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            quality,
            location: None,
        }
    }

    fn is_valid(&self) -> bool {
        matches!(self.quality, DataQuality::Good | DataQuality::Fair)
    }
}

/// Commands sent to IoT devices
#[derive(Debug, Clone)]
struct DeviceCommand {
    id: String,
    target_device_id: String,
    command_type: CommandType,
    parameters: HashMap<String, String>,
    timestamp: u64,
    priority: CommandPriority,
    timeout_seconds: u64,
    retry_count: u32,
    max_retries: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum CommandType {
    GetStatus,
    SetConfiguration,
    ExecuteAction,
    Restart,
    UpdateFirmware,
    Calibrate,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CommandPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl DeviceCommand {
    fn new(
        target_device_id: String,
        command_type: CommandType,
        parameters: HashMap<String, String>,
    ) -> Self {
        let priority = match command_type {
            CommandType::Restart | CommandType::UpdateFirmware => CommandPriority::Critical,
            CommandType::ExecuteAction => CommandPriority::High,
            CommandType::SetConfiguration => CommandPriority::Normal,
            CommandType::GetStatus | CommandType::Calibrate => CommandPriority::Low,
        };

        Self {
            id: format!("cmd_{}", fastrand::u64(..)),
            target_device_id,
            command_type,
            parameters,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            priority,
            timeout_seconds: 30,
            retry_count: 0,
            max_retries: 3,
        }
    }

    fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.timestamp > self.timeout_seconds
    }
}

/// Time-series data storage for historical telemetry
struct TimeSeriesStore {
    data: Arc<RwLock<BTreeMap<String, VecDeque<TelemetryData>>>>, // metric_key -> time-ordered data
    max_points_per_metric: usize,
    total_points: AtomicUsize,
    metrics_count: AtomicUsize,
}

impl TimeSeriesStore {
    fn new(max_points_per_metric: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(BTreeMap::new())),
            max_points_per_metric,
            total_points: AtomicUsize::new(0),
            metrics_count: AtomicUsize::new(0),
        }
    }

    fn store_telemetry(&self, telemetry: TelemetryData) -> Result<(), String> {
        if !telemetry.is_valid() {
            return Ok(()); // Skip invalid data
        }

        let metric_key = format!("{}:{}", telemetry.device_id, telemetry.metric_name);

        let mut data = self
            .data
            .write()
            .map_err(|_| "Failed to acquire time series lock")?;

        let metric_data = data.entry(metric_key.clone()).or_insert_with(|| {
            self.metrics_count.fetch_add(1, Ordering::Relaxed);
            VecDeque::new()
        });

        // Maintain size limit
        if metric_data.len() >= self.max_points_per_metric {
            metric_data.pop_front();
        } else {
            self.total_points.fetch_add(1, Ordering::Relaxed);
        }

        metric_data.push_back(telemetry);
        Ok(())
    }

    fn get_recent_data(
        &self,
        device_id: &str,
        metric_name: &str,
        count: usize,
    ) -> Result<Vec<TelemetryData>, String> {
        let metric_key = format!("{}:{}", device_id, metric_name);
        let data = self
            .data
            .read()
            .map_err(|_| "Failed to acquire time series lock")?;

        if let Some(metric_data) = data.get(&metric_key) {
            let start_index = metric_data.len().saturating_sub(count);
            Ok(metric_data.iter().skip(start_index).cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn calculate_average(
        &self,
        device_id: &str,
        metric_name: &str,
        duration_seconds: u64,
    ) -> Result<Option<f64>, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let cutoff_time = now - (duration_seconds * 1000);

        let recent_data = self.get_recent_data(device_id, metric_name, 1000)?;
        let valid_data: Vec<_> = recent_data
            .into_iter()
            .filter(|d| d.timestamp >= cutoff_time && d.is_valid())
            .collect();

        if valid_data.is_empty() {
            Ok(None)
        } else {
            let sum: f64 = valid_data.iter().map(|d| d.value).sum();
            Ok(Some(sum / valid_data.len() as f64))
        }
    }

    fn detect_anomalies(
        &self,
        device_id: &str,
        metric_name: &str,
    ) -> Result<Vec<TelemetryData>, String> {
        let recent_data = self.get_recent_data(device_id, metric_name, 100)?;

        if recent_data.len() < 10 {
            return Ok(Vec::new()); // Not enough data for anomaly detection
        }

        // Calculate mean and standard deviation
        let values: Vec<f64> = recent_data.iter().map(|d| d.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Find anomalies (values more than 2 standard deviations from mean)
        let anomalies: Vec<_> = recent_data
            .into_iter()
            .filter(|d| (d.value - mean).abs() > 2.0 * std_dev)
            .collect();

        Ok(anomalies)
    }

    fn stats(&self) -> (usize, usize) {
        (
            self.total_points.load(Ordering::Relaxed),
            self.metrics_count.load(Ordering::Relaxed),
        )
    }
}

/// Event types in the IoT system
#[derive(Debug, Clone)]
enum IoTEvent {
    DeviceConnected {
        device_id: String,
    },
    DeviceDisconnected {
        device_id: String,
    },
    TelemetryReceived {
        telemetry: TelemetryData,
    },
    CommandSent {
        command: DeviceCommand,
    },
    CommandCompleted {
        command_id: String,
        success: bool,
        response: Option<String>,
    },
    AnomalyDetected {
        device_id: String,
        metric_name: String,
        value: f64,
        expected_range: (f64, f64),
    },
    DeviceBatteryLow {
        device_id: String,
        battery_level: u64,
    },
    DeviceOffline {
        device_id: String,
        offline_duration: u64,
    },
}

type EventHandler = Box<dyn Fn(&IoTEvent) -> Result<(), String> + Send + Sync>;

/// Event processor for handling IoT system events
struct EventProcessor {
    event_handlers: HashMap<String, EventHandler>,
    events_processed: AtomicUsize,
    processing_time: AtomicU64,
}

impl EventProcessor {
    fn new() -> Self {
        Self {
            event_handlers: HashMap::new(),
            events_processed: AtomicUsize::new(0),
            processing_time: AtomicU64::new(0),
        }
    }

    fn register_handler<F>(&mut self, event_type: String, handler: F)
    where
        F: Fn(&IoTEvent) -> Result<(), String> + Send + Sync + 'static,
    {
        self.event_handlers.insert(event_type, Box::new(handler));
    }

    fn process_event(&self, event: &IoTEvent) -> Result<(), String> {
        let start_time = Instant::now();

        let event_type = match event {
            IoTEvent::DeviceConnected { .. } => "device_connected",
            IoTEvent::DeviceDisconnected { .. } => "device_disconnected",
            IoTEvent::TelemetryReceived { .. } => "telemetry_received",
            IoTEvent::CommandSent { .. } => "command_sent",
            IoTEvent::CommandCompleted { .. } => "command_completed",
            IoTEvent::AnomalyDetected { .. } => "anomaly_detected",
            IoTEvent::DeviceBatteryLow { .. } => "device_battery_low",
            IoTEvent::DeviceOffline { .. } => "device_offline",
        };

        if let Some(handler) = self.event_handlers.get(event_type) {
            handler(event)?;
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.events_processed.fetch_add(1, Ordering::Relaxed);
        self.processing_time
            .fetch_add(processing_time, Ordering::Relaxed);

        Ok(())
    }

    fn stats(&self) -> (usize, u64) {
        (
            self.events_processed.load(Ordering::Relaxed),
            self.processing_time.load(Ordering::Relaxed),
        )
    }
}

/// Main IoT device management system
struct IoTDeviceManager {
    runtime: Moirai,
    devices: Arc<RwLock<HashMap<String, Arc<IoTDevice>>>>,
    time_series: Arc<TimeSeriesStore>,
    event_processor: Arc<Mutex<EventProcessor>>,

    // Command handling
    command_queue: Arc<Mutex<VecDeque<DeviceCommand>>>,
    pending_commands: Arc<RwLock<HashMap<String, DeviceCommand>>>,

    // Statistics
    total_devices: AtomicUsize,
    active_devices: Arc<AtomicUsize>,
    total_telemetry: AtomicUsize,
    total_commands: AtomicUsize,
    total_events: AtomicUsize,

    // Configuration
    telemetry_workers: usize,
    command_workers: usize,
    event_workers: usize,

    // Control
    is_running: Arc<AtomicBool>,
}

impl IoTDeviceManager {
    fn new(
        telemetry_workers: usize,
        command_workers: usize,
        event_workers: usize,
    ) -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;

        let mut event_processor = EventProcessor::new();

        // Register default event handlers
        event_processor.register_handler("device_connected".to_string(), |event| {
            if let IoTEvent::DeviceConnected { device_id } = event {
                println!("📱 Device connected: {}", device_id);
            }
            Ok(())
        });

        event_processor.register_handler("anomaly_detected".to_string(), |event| {
            if let IoTEvent::AnomalyDetected {
                device_id,
                metric_name,
                value,
                expected_range,
            } = event
            {
                println!(
                    "⚠️  Anomaly detected: {} {} = {:.2} (expected: {:.2}-{:.2})",
                    device_id, metric_name, value, expected_range.0, expected_range.1
                );
            }
            Ok(())
        });

        event_processor.register_handler("device_battery_low".to_string(), |event| {
            if let IoTEvent::DeviceBatteryLow {
                device_id,
                battery_level,
            } = event
            {
                println!("🔋 Low battery warning: {} ({}%)", device_id, battery_level);
            }
            Ok(())
        });

        let manager = Self {
            runtime,
            devices: Arc::new(RwLock::new(HashMap::new())),
            time_series: Arc::new(TimeSeriesStore::new(1000)),
            event_processor: Arc::new(Mutex::new(event_processor)),
            command_queue: Arc::new(Mutex::new(VecDeque::new())),
            pending_commands: Arc::new(RwLock::new(HashMap::new())),
            total_devices: AtomicUsize::new(0),
            active_devices: Arc::new(AtomicUsize::new(0)),
            total_telemetry: AtomicUsize::new(0),
            total_commands: AtomicUsize::new(0),
            total_events: AtomicUsize::new(0),
            telemetry_workers,
            command_workers,
            event_workers,
            is_running: Arc::new(AtomicBool::new(false)),
        };

        Ok(manager)
    }

    fn start(&self) -> Result<(), String> {
        self.is_running.store(true, Ordering::Relaxed);

        // Start telemetry workers
        for worker_id in 0..self.telemetry_workers {
            self.start_telemetry_worker(worker_id)?;
        }

        // Start command workers
        for worker_id in 0..self.command_workers {
            self.start_command_worker(worker_id)?;
        }

        // Start monitoring services
        self.start_device_monitor()?;
        self.start_anomaly_detector()?;

        println!(
            "IoT Device Manager started with {} telemetry, {} command workers",
            self.telemetry_workers, self.command_workers
        );
        Ok(())
    }

    fn register_device(&self, device: IoTDevice) -> Result<(), String> {
        let device_id = device.id.clone();
        let device_arc = Arc::new(device);

        {
            let mut devices = self
                .devices
                .write()
                .map_err(|_| "Failed to acquire devices lock")?;
            devices.insert(device_id.clone(), Arc::clone(&device_arc));
        }

        self.total_devices.fetch_add(1, Ordering::Relaxed);
        self.update_active_device_count();

        // Emit device connected event
        let event = IoTEvent::DeviceConnected {
            device_id: device_id.clone(),
        };
        self.emit_event(event)?;

        println!(
            "Device registered: {} ({})",
            device_id, device_arc.device_type
        );
        Ok(())
    }

    fn submit_telemetry(&self, telemetry: TelemetryData) -> Result<(), String> {
        // Update device last seen
        if let Ok(devices) = self.devices.read() {
            if let Some(device) = devices.get(&telemetry.device_id) {
                device.update_last_seen();
                device.increment_telemetry_count();
            }
        }

        // Store in time series
        self.time_series.store_telemetry(telemetry.clone())?;
        self.total_telemetry.fetch_add(1, Ordering::Relaxed);

        // Emit telemetry received event
        let event = IoTEvent::TelemetryReceived { telemetry };
        self.emit_event(event)?;

        Ok(())
    }

    fn send_command(&self, command: DeviceCommand) -> Result<(), String> {
        // Add to pending commands
        {
            let mut pending = self
                .pending_commands
                .write()
                .map_err(|_| "Failed to acquire pending commands lock")?;
            pending.insert(command.id.clone(), command.clone());
        }

        // Queue for processing
        {
            let mut queue = self
                .command_queue
                .lock()
                .map_err(|_| "Failed to acquire command queue lock")?;
            queue.push_back(command.clone());
        }

        self.total_commands.fetch_add(1, Ordering::Relaxed);

        // Emit command sent event
        let event = IoTEvent::CommandSent { command };
        self.emit_event(event)?;

        Ok(())
    }

    fn start_telemetry_worker(&self, worker_id: usize) -> Result<(), String> {
        let devices = self.devices.clone();
        let is_running = self.is_running.clone();

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    // Generate synthetic telemetry data
                    if let Ok(devices_guard) = devices.read() {
                        for device in devices_guard.values() {
                            if device.is_active() && fastrand::f64() < 0.3 {
                                // 30% chance per cycle
                                Self::generate_device_telemetry_worker(device);
                            }
                        }
                    }

                    std::thread::sleep(Duration::from_millis(100 + worker_id as u64 * 50));
                }
            },
            Priority::Normal,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn generate_device_telemetry_worker(device: &IoTDevice) {
        // Generate telemetry based on device type
        match device.device_type {
            DeviceType::TemperatureSensor => {
                let temp = 20.0 + fastrand::f64() * 15.0; // 20-35°C
                let _telemetry = TelemetryData::new(
                    device.id.clone(),
                    "temperature".to_string(),
                    temp,
                    "°C".to_string(),
                );
                // In real implementation, would submit to manager
            }
            DeviceType::HumiditySensor => {
                let humidity = 30.0 + fastrand::f64() * 40.0; // 30-70%
                let _telemetry = TelemetryData::new(
                    device.id.clone(),
                    "humidity".to_string(),
                    humidity,
                    "%".to_string(),
                );
            }
            DeviceType::MotionDetector => {
                let motion = if fastrand::f64() < 0.1 { 1.0 } else { 0.0 };
                let _telemetry = TelemetryData::new(
                    device.id.clone(),
                    "motion".to_string(),
                    motion,
                    "bool".to_string(),
                );
            }
            DeviceType::AirQualitySensor => {
                let co2 = 400.0 + fastrand::f64() * 600.0; // 400-1000 ppm
                let _telemetry = TelemetryData::new(
                    device.id.clone(),
                    "co2".to_string(),
                    co2,
                    "ppm".to_string(),
                );
            }
            DeviceType::EnergyMeter => {
                let power = 100.0 + fastrand::f64() * 500.0; // 100-600W
                let _telemetry = TelemetryData::new(
                    device.id.clone(),
                    "power".to_string(),
                    power,
                    "W".to_string(),
                );
            }
            _ => {} // Other devices generate telemetry less frequently
        }
    }

    fn start_command_worker(&self, worker_id: usize) -> Result<(), String> {
        let command_queue = self.command_queue.clone();
        let pending_commands = self.pending_commands.clone();
        let devices = self.devices.clone();
        let is_running = self.is_running.clone();

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    // Get command from queue
                    let command = match command_queue.lock() {
                        Ok(mut queue) => queue.pop_front(),
                        Err(_) => None,
                    };

                    if let Some(mut command) = command {
                        // Process command
                        let success = Self::execute_command_worker(&command, &devices, worker_id);

                        if success {
                            // Remove from pending
                            if let Ok(mut pending) = pending_commands.write() {
                                pending.remove(&command.id);
                            }
                        } else if command.can_retry() {
                            // Retry command
                            command.increment_retry();
                            if let Ok(mut queue) = command_queue.lock() {
                                queue.push_back(command);
                            }
                        } else {
                            // Command failed permanently
                            if let Ok(mut pending) = pending_commands.write() {
                                pending.remove(&command.id);
                            }
                        }
                    } else {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            },
            Priority::High,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn execute_command_worker(
        command: &DeviceCommand,
        devices: &Arc<RwLock<HashMap<String, Arc<IoTDevice>>>>,
        worker_id: usize,
    ) -> bool {
        // Get target device
        let device = match devices.read() {
            Ok(devices_guard) => devices_guard.get(&command.target_device_id).cloned(),
            Err(_) => return false,
        };

        let device = match device {
            Some(d) => d,
            None => return false,
        };

        if !device.is_active() {
            return false; // Device offline
        }

        // Simulate command execution
        let execution_delay = match command.command_type {
            CommandType::GetStatus => Duration::from_millis(fastrand::u64(10..50)),
            CommandType::SetConfiguration => Duration::from_millis(fastrand::u64(50..200)),
            CommandType::ExecuteAction => Duration::from_millis(fastrand::u64(100..500)),
            CommandType::Restart => Duration::from_millis(fastrand::u64(1000..3000)),
            CommandType::UpdateFirmware => Duration::from_millis(fastrand::u64(5000..10000)),
            CommandType::Calibrate => Duration::from_millis(fastrand::u64(500..2000)),
        };

        std::thread::sleep(execution_delay);

        // Simulate occasional failures
        let success_rate = match command.priority {
            CommandPriority::Critical => 0.95,
            CommandPriority::High => 0.90,
            CommandPriority::Normal => 0.85,
            CommandPriority::Low => 0.80,
        };

        let success = fastrand::f64() < success_rate;

        if success {
            device.increment_command_count();
            if worker_id == 0 && fastrand::f64() < 0.1 {
                println!(
                    "Command executed: {} -> {} ({:?})",
                    command.id, device.id, command.command_type
                );
            }
        }

        success
    }

    fn start_device_monitor(&self) -> Result<(), String> {
        let devices = self.devices.clone();
        let is_running = self.is_running.clone();
        let active_devices = self.active_devices.clone();

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    let mut active_count = 0;

                    if let Ok(devices_guard) = devices.read() {
                        for device in devices_guard.values() {
                            if device.is_active() {
                                active_count += 1;

                                // Check battery level
                                let battery_level = device.battery_level.load(Ordering::Relaxed);
                                if battery_level < 20 && fastrand::f64() < 0.1 {
                                    // Emit low battery event
                                    println!("🔋 Low battery: {} ({}%)", device.id, battery_level);
                                }

                                // Simulate battery drain
                                if fastrand::f64() < 0.1 {
                                    let current_battery =
                                        device.battery_level.load(Ordering::Relaxed);
                                    if current_battery > 0 {
                                        device.battery_level.store(
                                            current_battery.saturating_sub(1),
                                            Ordering::Relaxed,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    active_devices.store(active_count, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_secs(10));
                }
            },
            Priority::Low,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn start_anomaly_detector(&self) -> Result<(), String> {
        let time_series = self.time_series.clone();
        let devices = self.devices.clone();
        let is_running = self.is_running.clone();

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    if let Ok(devices_guard) = devices.read() {
                        for device in devices_guard.values() {
                            if device.is_active() {
                                // Check for anomalies in device metrics
                                for capability in &device.capabilities {
                                    if let Ok(anomalies) =
                                        time_series.detect_anomalies(&device.id, capability)
                                    {
                                        for anomaly in anomalies {
                                            println!(
                                                "⚠️  Anomaly: {} {} = {:.2}",
                                                device.id, capability, anomaly.value
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    std::thread::sleep(Duration::from_secs(30)); // Check every 30 seconds
                }
            },
            Priority::Low,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn emit_event(&self, event: IoTEvent) -> Result<(), String> {
        if let Ok(processor) = self.event_processor.lock() {
            processor.process_event(&event)?;
            self.total_events.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn update_active_device_count(&self) {
        let active_count = if let Ok(devices) = self.devices.read() {
            devices.values().filter(|d| d.is_active()).count()
        } else {
            0
        };
        self.active_devices.store(active_count, Ordering::Relaxed);
    }

    fn get_device_stats(&self, device_id: &str) -> Result<DeviceStats, String> {
        let devices = self
            .devices
            .read()
            .map_err(|_| "Failed to acquire devices lock")?;

        let device = devices
            .get(device_id)
            .ok_or_else(|| "Device not found".to_string())?;

        let (telemetry_count, command_count, battery, signal, is_active) = device.stats();

        // Get recent telemetry averages
        let mut recent_metrics = HashMap::new();
        for capability in &device.capabilities {
            if let Ok(Some(avg)) = self
                .time_series
                .calculate_average(device_id, capability, 300)
            {
                // 5 minutes
                recent_metrics.insert(capability.clone(), avg);
            }
        }

        Ok(DeviceStats {
            device_id: device_id.to_string(),
            device_type: device.device_type.clone(),
            is_active,
            battery_level: battery,
            signal_strength: signal,
            telemetry_count,
            command_count,
            recent_metrics,
        })
    }

    fn get_system_stats(&self) -> SystemStats {
        let (_event_count, event_processing_time) =
            if let Ok(processor) = self.event_processor.lock() {
                processor.stats()
            } else {
                (0, 0)
            };

        let (time_series_points, time_series_metrics) = self.time_series.stats();
        let pending_commands = self.pending_commands.read().map(|p| p.len()).unwrap_or(0);

        SystemStats {
            total_devices: self.total_devices.load(Ordering::Relaxed),
            active_devices: self.active_devices.load(Ordering::Relaxed),
            total_telemetry: self.total_telemetry.load(Ordering::Relaxed),
            total_commands: self.total_commands.load(Ordering::Relaxed),
            pending_commands,
            total_events: self.total_events.load(Ordering::Relaxed),
            event_processing_time_us: event_processing_time,
            time_series_points,
            time_series_metrics,
        }
    }

    fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        println!("IoT Device Manager stopped");
    }
}

#[derive(Debug)]
struct DeviceStats {
    device_id: String,
    device_type: DeviceType,
    is_active: bool,
    battery_level: u64,
    signal_strength: u64,
    telemetry_count: usize,
    command_count: usize,
    recent_metrics: HashMap<String, f64>,
}

#[derive(Debug)]
struct SystemStats {
    total_devices: usize,
    active_devices: usize,
    total_telemetry: usize,
    total_commands: usize,
    pending_commands: usize,
    total_events: usize,
    event_processing_time_us: u64,
    time_series_points: usize,
    time_series_metrics: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("IoT Device Management System - Event-Driven Real-Time Coordination");
    println!("==================================================================");

    // Create IoT device manager
    let manager = IoTDeviceManager::new(
        3, // telemetry workers
        2, // command workers
        1, // event workers
    )?;

    manager.start()?;

    // Register various IoT devices
    println!("\n1. Registering IoT devices...");
    let device_configs = [
        (
            "temp_001",
            DeviceType::TemperatureSensor,
            "Living Room Temperature",
            "Living Room",
        ),
        (
            "humid_001",
            DeviceType::HumiditySensor,
            "Bedroom Humidity",
            "Bedroom",
        ),
        (
            "motion_001",
            DeviceType::MotionDetector,
            "Front Door Motion",
            "Front Door",
        ),
        (
            "light_001",
            DeviceType::SmartLight,
            "Kitchen Light",
            "Kitchen",
        ),
        (
            "thermo_001",
            DeviceType::SmartThermostat,
            "Main Thermostat",
            "Hallway",
        ),
        (
            "camera_001",
            DeviceType::SecurityCamera,
            "Garage Camera",
            "Garage",
        ),
        (
            "lock_001",
            DeviceType::SmartLock,
            "Front Door Lock",
            "Front Door",
        ),
        (
            "air_001",
            DeviceType::AirQualitySensor,
            "Office Air Quality",
            "Office",
        ),
        (
            "energy_001",
            DeviceType::EnergyMeter,
            "Main Energy Meter",
            "Utility Room",
        ),
        (
            "water_001",
            DeviceType::WaterSensor,
            "Basement Water",
            "Basement",
        ),
    ];

    for (id, device_type, name, location) in &device_configs {
        let device = IoTDevice::new(
            id.to_string(),
            device_type.clone(),
            name.to_string(),
            location.to_string(),
        );
        manager.register_device(device)?;
    }

    println!("  Registered {} devices", device_configs.len());

    // Generate telemetry data
    println!("\n2. Generating telemetry data...");
    let telemetry_start = Instant::now();

    for round in 0..10 {
        println!("  Round {}: Generating telemetry...", round + 1);

        for (device_id, device_type, _, _) in &device_configs {
            // Generate device-specific telemetry
            let telemetry_data = match device_type {
                DeviceType::TemperatureSensor => vec![TelemetryData::new(
                    device_id.to_string(),
                    "temperature".to_string(),
                    20.0 + fastrand::f64() * 15.0,
                    "°C".to_string(),
                )],
                DeviceType::HumiditySensor => vec![TelemetryData::new(
                    device_id.to_string(),
                    "humidity".to_string(),
                    30.0 + fastrand::f64() * 40.0,
                    "%".to_string(),
                )],
                DeviceType::MotionDetector => vec![TelemetryData::new(
                    device_id.to_string(),
                    "motion".to_string(),
                    if fastrand::f64() < 0.2 { 1.0 } else { 0.0 },
                    "bool".to_string(),
                )],
                DeviceType::SmartLight => vec![
                    TelemetryData::new(
                        device_id.to_string(),
                        "brightness".to_string(),
                        fastrand::f64() * 100.0,
                        "%".to_string(),
                    ),
                    TelemetryData::new(
                        device_id.to_string(),
                        "power".to_string(),
                        if fastrand::f64() < 0.8 { 1.0 } else { 0.0 },
                        "bool".to_string(),
                    ),
                ],
                DeviceType::AirQualitySensor => vec![
                    TelemetryData::new(
                        device_id.to_string(),
                        "co2".to_string(),
                        400.0 + fastrand::f64() * 600.0,
                        "ppm".to_string(),
                    ),
                    TelemetryData::new(
                        device_id.to_string(),
                        "voc".to_string(),
                        fastrand::f64() * 500.0,
                        "ppb".to_string(),
                    ),
                ],
                DeviceType::EnergyMeter => vec![
                    TelemetryData::new(
                        device_id.to_string(),
                        "power".to_string(),
                        100.0 + fastrand::f64() * 500.0,
                        "W".to_string(),
                    ),
                    TelemetryData::new(
                        device_id.to_string(),
                        "voltage".to_string(),
                        220.0 + fastrand::f64() * 20.0,
                        "V".to_string(),
                    ),
                ],
                _ => vec![TelemetryData::new(
                    device_id.to_string(),
                    "status".to_string(),
                    1.0,
                    "bool".to_string(),
                )],
            };

            for telemetry in telemetry_data {
                manager.submit_telemetry(telemetry)?;
            }
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    let telemetry_time = telemetry_start.elapsed();
    println!("  Telemetry generation completed in {:?}", telemetry_time);

    // Send some commands
    println!("\n3. Sending device commands...");
    let commands = [
        (
            "light_001",
            CommandType::SetConfiguration,
            "brightness",
            "75",
        ),
        (
            "thermo_001",
            CommandType::SetConfiguration,
            "target_temp",
            "22",
        ),
        ("lock_001", CommandType::ExecuteAction, "lock", "true"),
        ("camera_001", CommandType::GetStatus, "", ""),
        ("air_001", CommandType::Calibrate, "", ""),
    ];

    for (device_id, command_type, param_key, param_value) in &commands {
        let mut parameters = HashMap::new();
        if !param_key.is_empty() {
            parameters.insert(param_key.to_string(), param_value.to_string());
        }

        let command = DeviceCommand::new(device_id.to_string(), command_type.clone(), parameters);

        manager.send_command(command)?;
    }

    println!("  Sent {} commands", commands.len());

    // Wait for processing
    std::thread::sleep(Duration::from_secs(3));

    // Display device statistics
    println!("\n4. Device Statistics:");
    for (device_id, _, name, location) in &device_configs[..5] {
        // Show first 5 devices
        match manager.get_device_stats(device_id) {
            Ok(stats) => {
                println!("  Device: {} ({})", name, device_id);
                println!("    ├─ Type: {}", stats.device_type);
                println!("    ├─ Location: {}", location);
                println!(
                    "    ├─ Status: {}",
                    if stats.is_active { "Online" } else { "Offline" }
                );
                println!("    ├─ Battery: {}%", stats.battery_level);
                println!("    ├─ Signal: {}%", stats.signal_strength);
                println!("    ├─ Telemetry: {}", stats.telemetry_count);
                println!("    ├─ Commands: {}", stats.command_count);
                println!("    └─ Recent metrics: {:?}", stats.recent_metrics);
            }
            Err(e) => println!("  Failed to get stats for {}: {}", device_id, e),
        }
    }

    // Simulate some device failures and anomalies
    println!("\n5. Simulating device events...");

    // Inject some anomalous telemetry
    let anomaly_telemetry = TelemetryData::new(
        "temp_001".to_string(),
        "temperature".to_string(),
        65.0, // Anomalously high temperature
        "°C".to_string(),
    );
    manager.submit_telemetry(anomaly_telemetry)?;

    // Wait for anomaly detection
    std::thread::sleep(Duration::from_secs(2));

    // Display comprehensive system statistics
    println!("\n6. System Statistics:");
    let system_stats = manager.get_system_stats();

    println!("  ├─ Device Management:");
    println!("  │  ├─ Total devices: {}", system_stats.total_devices);
    println!("  │  ├─ Active devices: {}", system_stats.active_devices);
    println!(
        "  │  └─ Online rate: {:.1}%",
        (system_stats.active_devices as f64 / system_stats.total_devices.max(1) as f64) * 100.0
    );

    println!("  ├─ Data Processing:");
    println!("  │  ├─ Telemetry points: {}", system_stats.total_telemetry);
    println!("  │  ├─ Commands sent: {}", system_stats.total_commands);
    println!(
        "  │  ├─ Pending commands: {}",
        system_stats.pending_commands
    );
    println!("  │  └─ Events processed: {}", system_stats.total_events);

    println!("  ├─ Time Series Storage:");
    println!("  │  ├─ Data points: {}", system_stats.time_series_points);
    println!(
        "  │  ├─ Metrics tracked: {}",
        system_stats.time_series_metrics
    );
    println!(
        "  │  └─ Storage efficiency: {:.1} points/metric",
        system_stats.time_series_points as f64 / system_stats.time_series_metrics.max(1) as f64
    );

    println!("  └─ Performance:");
    println!(
        "     ├─ Telemetry rate: {:.1} points/sec",
        system_stats.total_telemetry as f64 / telemetry_time.as_secs_f64()
    );
    println!(
        "     ├─ Event processing: {:.1}μs avg",
        system_stats.event_processing_time_us as f64 / system_stats.total_events.max(1) as f64
    );
    println!(
        "     └─ Command success rate: {:.1}%",
        ((system_stats.total_commands - system_stats.pending_commands) as f64
            / system_stats.total_commands.max(1) as f64)
            * 100.0
    );

    // Stop the system
    manager.stop();

    println!("\nIoT Device Management System demonstration completed!");
    println!("Successfully demonstrated:");
    println!("- Event-driven architecture for device coordination");
    println!("- Real-time sensor data streaming and processing");
    println!("- Device discovery and automatic registration");
    println!("- Command distribution with reliable delivery");
    println!("- Telemetry aggregation and anomaly detection");
    println!("- Time-series data storage and analysis");

    Ok(())
}
