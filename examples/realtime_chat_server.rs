//! Real-time Chat Server - WebSocket Pub/Sub with Concurrent Message Delivery
//!
//! This example demonstrates:
//! - WebSocket-style real-time message handling with async patterns
//! - Publish/subscribe system with topic-based routing
//! - Concurrent user session management with heartbeat monitoring
//! - Message queuing with priority and delivery guarantees
//! - Real-time presence tracking and room management
//! - Event-driven architecture with message broadcasting

use moirai::{Moirai, Priority};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::fmt;

/// Represents a user in the chat system
#[derive(Debug, Clone)]
struct User {
    id: u64,
    username: String,
    session_id: String,
    connected_at: u64,
    last_seen: AtomicU64,
    is_online: AtomicBool,
    message_count: AtomicUsize,
    rooms: Arc<RwLock<HashSet<String>>>,
}

impl User {
    fn new(id: u64, username: String, session_id: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            id,
            username,
            session_id,
            connected_at: now,
            last_seen: AtomicU64::new(now),
            is_online: AtomicBool::new(true),
            message_count: AtomicUsize::new(0),
            rooms: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    fn update_activity(&self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.last_seen.store(now, Ordering::Relaxed);
        self.is_online.store(true, Ordering::Relaxed);
    }

    fn increment_message_count(&self) {
        self.message_count.fetch_add(1, Ordering::Relaxed);
    }

    fn join_room(&self, room_name: &str) -> Result<(), String> {
        let mut rooms = self.rooms.write()
            .map_err(|_| "Failed to acquire rooms lock")?;
        rooms.insert(room_name.to_string());
        Ok(())
    }

    fn leave_room(&self, room_name: &str) -> Result<(), String> {
        let mut rooms = self.rooms.write()
            .map_err(|_| "Failed to acquire rooms lock")?;
        rooms.remove(room_name);
        Ok(())
    }

    fn is_in_room(&self, room_name: &str) -> bool {
        self.rooms.read()
            .map(|rooms| rooms.contains(room_name))
            .unwrap_or(false)
    }

    fn get_rooms(&self) -> Vec<String> {
        self.rooms.read()
            .map(|rooms| rooms.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn is_active(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let last_seen = self.last_seen.load(Ordering::Relaxed);
        self.is_online.load(Ordering::Relaxed) && (now - last_seen) < 300 // 5 minutes
    }
}

/// Different types of messages in the chat system
#[derive(Debug, Clone)]
enum MessageType {
    Text,
    Image,
    File,
    System,
    Heartbeat,
    TypingIndicator,
    Reaction,
}

impl fmt::Display for MessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageType::Text => write!(f, "TEXT"),
            MessageType::Image => write!(f, "IMAGE"),
            MessageType::File => write!(f, "FILE"),
            MessageType::System => write!(f, "SYSTEM"),
            MessageType::Heartbeat => write!(f, "HEARTBEAT"),
            MessageType::TypingIndicator => write!(f, "TYPING"),
            MessageType::Reaction => write!(f, "REACTION"),
        }
    }
}

/// Chat message with metadata
#[derive(Debug, Clone)]
struct Message {
    id: u64,
    sender_id: u64,
    sender_username: String,
    room_name: String,
    message_type: MessageType,
    content: String,
    timestamp: u64,
    priority: MessagePriority,
    delivery_attempts: u32,
    max_delivery_attempts: u32,
    expiry_time: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum MessagePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    System = 4,
}

impl Message {
    fn new(sender_id: u64, sender_username: String, room_name: String, message_type: MessageType, content: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let priority = match message_type {
            MessageType::System => MessagePriority::System,
            MessageType::Heartbeat => MessagePriority::Low,
            MessageType::TypingIndicator => MessagePriority::Low,
            _ => MessagePriority::Normal,
        };

        Self {
            id: fastrand::u64(..),
            sender_id,
            sender_username,
            room_name,
            message_type,
            content,
            timestamp: now,
            priority,
            delivery_attempts: 0,
            max_delivery_attempts: 3,
            expiry_time: if matches!(message_type, MessageType::TypingIndicator) {
                Some(now + 10) // Typing indicators expire after 10 seconds
            } else {
                None
            },
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(expiry) = self.expiry_time {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            now > expiry
        } else {
            false
        }
    }

    fn can_retry(&self) -> bool {
        self.delivery_attempts < self.max_delivery_attempts && !self.is_expired()
    }

    fn increment_delivery_attempt(&mut self) {
        self.delivery_attempts += 1;
    }
}

/// Chat room with user management and message history
#[derive(Debug)]
struct ChatRoom {
    name: String,
    topic: Option<String>,
    created_at: u64,
    users: Arc<RwLock<HashSet<u64>>>,
    message_history: Arc<Mutex<VecDeque<Message>>>,
    max_history_size: usize,
    total_messages: AtomicUsize,
    active_users: AtomicUsize,
    typing_users: Arc<RwLock<HashMap<u64, u64>>>, // user_id -> timestamp
}

impl ChatRoom {
    fn new(name: String, topic: Option<String>) -> Self {
        Self {
            name,
            topic,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            users: Arc::new(RwLock::new(HashSet::new())),
            message_history: Arc::new(Mutex::new(VecDeque::new())),
            max_history_size: 1000,
            total_messages: AtomicUsize::new(0),
            active_users: AtomicUsize::new(0),
            typing_users: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn add_user(&self, user_id: u64) -> Result<bool, String> {
        let mut users = self.users.write()
            .map_err(|_| "Failed to acquire users lock")?;
        let was_new = users.insert(user_id);
        if was_new {
            self.active_users.store(users.len(), Ordering::Relaxed);
        }
        Ok(was_new)
    }

    fn remove_user(&self, user_id: u64) -> Result<bool, String> {
        let mut users = self.users.write()
            .map_err(|_| "Failed to acquire users lock")?;
        let was_present = users.remove(&user_id);
        if was_present {
            self.active_users.store(users.len(), Ordering::Relaxed);
            
            // Remove from typing users
            if let Ok(mut typing) = self.typing_users.write() {
                typing.remove(&user_id);
            }
        }
        Ok(was_present)
    }

    fn add_message(&self, message: Message) -> Result<(), String> {
        let mut history = self.message_history.lock()
            .map_err(|_| "Failed to acquire message history lock")?;
        
        // Maintain history size limit
        if history.len() >= self.max_history_size {
            history.pop_front();
        }
        
        history.push_back(message);
        self.total_messages.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn get_recent_messages(&self, count: usize) -> Result<Vec<Message>, String> {
        let history = self.message_history.lock()
            .map_err(|_| "Failed to acquire message history lock")?;
        
        let start_index = history.len().saturating_sub(count);
        Ok(history.iter().skip(start_index).cloned().collect())
    }

    fn update_typing_indicator(&self, user_id: u64, is_typing: bool) -> Result<(), String> {
        let mut typing = self.typing_users.write()
            .map_err(|_| "Failed to acquire typing users lock")?;
        
        if is_typing {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            typing.insert(user_id, now);
        } else {
            typing.remove(&user_id);
        }
        Ok(())
    }

    fn get_typing_users(&self) -> Vec<u64> {
        if let Ok(typing) = self.typing_users.read() {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            typing.iter()
                .filter(|(_, &timestamp)| now - timestamp < 10) // 10 second timeout
                .map(|(&user_id, _)| user_id)
                .collect()
        } else {
            Vec::new()
        }
    }

    fn get_user_list(&self) -> Vec<u64> {
        self.users.read()
            .map(|users| users.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn user_count(&self) -> usize {
        self.active_users.load(Ordering::Relaxed)
    }

    fn message_count(&self) -> usize {
        self.total_messages.load(Ordering::Relaxed)
    }
}

/// Message delivery queue with priority and retry logic
struct MessageQueue {
    high_priority: Arc<Mutex<VecDeque<Message>>>,
    normal_priority: Arc<Mutex<VecDeque<Message>>>,
    low_priority: Arc<Mutex<VecDeque<Message>>>,
    retry_queue: Arc<Mutex<VecDeque<Message>>>,
    pending_deliveries: AtomicUsize,
    total_queued: AtomicUsize,
    successful_deliveries: AtomicUsize,
    failed_deliveries: AtomicUsize,
    delivery_latency: AtomicU64,
}

impl MessageQueue {
    fn new() -> Self {
        Self {
            high_priority: Arc::new(Mutex::new(VecDeque::new())),
            normal_priority: Arc::new(Mutex::new(VecDeque::new())),
            low_priority: Arc::new(Mutex::new(VecDeque::new())),
            retry_queue: Arc::new(Mutex::new(VecDeque::new())),
            pending_deliveries: AtomicUsize::new(0),
            total_queued: AtomicUsize::new(0),
            successful_deliveries: AtomicUsize::new(0),
            failed_deliveries: AtomicUsize::new(0),
            delivery_latency: AtomicU64::new(0),
        }
    }

    fn enqueue(&self, message: Message) -> Result<(), String> {
        if message.is_expired() {
            return Ok(()); // Don't queue expired messages
        }

        let queue = match message.priority {
            MessagePriority::System => &self.high_priority,
            MessagePriority::High => &self.high_priority,
            MessagePriority::Normal => &self.normal_priority,
            MessagePriority::Low => &self.low_priority,
        };

        queue.lock()
            .map_err(|_| "Failed to acquire queue lock")?
            .push_back(message);

        self.total_queued.fetch_add(1, Ordering::Relaxed);
        self.pending_deliveries.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn dequeue(&self) -> Option<Message> {
        // Check high priority first
        if let Ok(mut queue) = self.high_priority.lock() {
            if let Some(message) = queue.pop_front() {
                if !message.is_expired() {
                    return Some(message);
                }
            }
        }

        // Check retry queue
        if let Ok(mut queue) = self.retry_queue.lock() {
            if let Some(message) = queue.pop_front() {
                if !message.is_expired() {
                    return Some(message);
                }
            }
        }

        // Check normal priority
        if let Ok(mut queue) = self.normal_priority.lock() {
            if let Some(message) = queue.pop_front() {
                if !message.is_expired() {
                    return Some(message);
                }
            }
        }

        // Check low priority
        if let Ok(mut queue) = self.low_priority.lock() {
            if let Some(message) = queue.pop_front() {
                if !message.is_expired() {
                    return Some(message);
                }
            }
        }

        None
    }

    fn mark_delivery_success(&self, delivery_time_ms: u64) {
        self.pending_deliveries.fetch_sub(1, Ordering::Relaxed);
        self.successful_deliveries.fetch_add(1, Ordering::Relaxed);
        
        // Update rolling average delivery latency
        let current_avg = self.delivery_latency.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0 {
            delivery_time_ms
        } else {
            (current_avg * 7 + delivery_time_ms) / 8
        };
        self.delivery_latency.store(new_avg, Ordering::Relaxed);
    }

    fn mark_delivery_failure(&self, mut message: Message) {
        self.pending_deliveries.fetch_sub(1, Ordering::Relaxed);
        
        message.increment_delivery_attempt();
        
        if message.can_retry() {
            // Add to retry queue
            if let Ok(mut retry_queue) = self.retry_queue.lock() {
                retry_queue.push_back(message);
                self.pending_deliveries.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.failed_deliveries.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn stats(&self) -> (usize, usize, usize, usize, u64) {
        (
            self.total_queued.load(Ordering::Relaxed),
            self.pending_deliveries.load(Ordering::Relaxed),
            self.successful_deliveries.load(Ordering::Relaxed),
            self.failed_deliveries.load(Ordering::Relaxed),
            self.delivery_latency.load(Ordering::Relaxed),
        )
    }
}

/// Presence tracker for monitoring user activity
struct PresenceTracker {
    user_activities: Arc<RwLock<HashMap<u64, u64>>>, // user_id -> last_activity
    offline_threshold_seconds: u64,
    cleanup_interval_seconds: u64,
}

impl PresenceTracker {
    fn new(offline_threshold: u64, cleanup_interval: u64) -> Self {
        Self {
            user_activities: Arc::new(RwLock::new(HashMap::new())),
            offline_threshold_seconds: offline_threshold,
            cleanup_interval_seconds: cleanup_interval,
        }
    }

    fn update_user_activity(&self, user_id: u64) -> Result<(), String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let mut activities = self.user_activities.write()
            .map_err(|_| "Failed to acquire activities lock")?;
        activities.insert(user_id, now);
        Ok(())
    }

    fn is_user_online(&self, user_id: u64) -> bool {
        if let Ok(activities) = self.user_activities.read() {
            if let Some(&last_activity) = activities.get(&user_id) {
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                return now - last_activity <= self.offline_threshold_seconds;
            }
        }
        false
    }

    fn get_online_users(&self) -> Vec<u64> {
        if let Ok(activities) = self.user_activities.read() {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            activities.iter()
                .filter(|(_, &last_activity)| now - last_activity <= self.offline_threshold_seconds)
                .map(|(&user_id, _)| user_id)
                .collect()
        } else {
            Vec::new()
        }
    }

    fn cleanup_offline_users(&self) -> Result<usize, String> {
        let mut activities = self.user_activities.write()
            .map_err(|_| "Failed to acquire activities lock")?;
        
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let initial_count = activities.len();
        
        activities.retain(|_, &mut last_activity| {
            now - last_activity <= self.offline_threshold_seconds * 2 // Keep for 2x threshold for grace period
        });
        
        Ok(initial_count - activities.len())
    }
}

/// Main chat server with real-time messaging capabilities
struct ChatServer {
    runtime: Moirai,
    users: Arc<RwLock<HashMap<u64, Arc<User>>>>,
    rooms: Arc<RwLock<HashMap<String, Arc<ChatRoom>>>>,
    message_queue: Arc<MessageQueue>,
    presence_tracker: Arc<PresenceTracker>,
    
    // Configuration
    max_users: usize,
    max_rooms: usize,
    message_workers: usize,
    
    // Statistics
    total_connections: AtomicUsize,
    active_connections: AtomicUsize,
    messages_sent: AtomicUsize,
    messages_received: AtomicUsize,
    server_start_time: u64,
    
    // State
    is_running: AtomicBool,
    next_user_id: AtomicU64,
}

impl ChatServer {
    fn new(max_users: usize, max_rooms: usize, message_workers: usize) -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;
        
        let server = Self {
            runtime,
            users: Arc::new(RwLock::new(HashMap::new())),
            rooms: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(MessageQueue::new()),
            presence_tracker: Arc::new(PresenceTracker::new(300, 60)), // 5 min offline, 1 min cleanup
            max_users,
            max_rooms,
            message_workers,
            total_connections: AtomicUsize::new(0),
            active_connections: AtomicUsize::new(0),
            messages_sent: AtomicUsize::new(0),
            messages_received: AtomicUsize::new(0),
            server_start_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            is_running: AtomicBool::new(false),
            next_user_id: AtomicU64::new(1),
        };

        server.start_background_services()?;
        Ok(server)
    }

    fn start(&self) -> Result<(), String> {
        self.is_running.store(true, Ordering::Relaxed);
        println!("Chat server started with {} message workers", self.message_workers);
        
        // Start message delivery workers
        for worker_id in 0..self.message_workers {
            self.start_message_worker(worker_id)?;
        }
        
        Ok(())
    }

    fn start_message_worker(&self, worker_id: usize) -> Result<(), String> {
        let message_queue = self.message_queue.clone();
        let users = self.users.clone();
        let rooms = self.rooms.clone();
        let is_running = self.is_running.clone();
        let messages_sent = self.messages_sent.clone();

        let handle = self.runtime.spawn_fn_with_priority(move || {
            while is_running.load(Ordering::Relaxed) {
                if let Some(message) = message_queue.dequeue() {
                    let delivery_start = Instant::now();
                    let success = Self::deliver_message_worker(&message, &users, &rooms);
                    let delivery_time = delivery_start.elapsed().as_millis() as u64;
                    
                    if success {
                        message_queue.mark_delivery_success(delivery_time);
                        messages_sent.fetch_add(1, Ordering::Relaxed);
                        
                        if worker_id == 0 && messages_sent.load(Ordering::Relaxed) % 100 == 0 {
                            println!("Worker {}: Delivered {} messages", worker_id, messages_sent.load(Ordering::Relaxed));
                        }
                    } else {
                        message_queue.mark_delivery_failure(message);
                    }
                } else {
                    // No messages to process, wait a bit
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }, Priority::High);

        std::mem::drop(handle);
        Ok(())
    }

    fn deliver_message_worker(
        message: &Message,
        users: &Arc<RwLock<HashMap<u64, Arc<User>>>>,
        rooms: &Arc<RwLock<HashMap<String, Arc<ChatRoom>>>>
    ) -> bool {
        // Get the room
        let room = {
            let rooms_guard = match rooms.read() {
                Ok(guard) => guard,
                Err(_) => return false,
            };
            
            match rooms_guard.get(&message.room_name) {
                Some(room) => room.clone(),
                None => return false,
            }
        };

        // Add message to room history
        if room.add_message(message.clone()).is_err() {
            return false;
        }

        // Get users in the room
        let room_users = room.get_user_list();
        
        // Simulate message delivery to each user
        let mut delivery_count = 0;
        for user_id in room_users {
            if user_id != message.sender_id { // Don't echo back to sender
                if let Ok(users_guard) = users.read() {
                    if let Some(user) = users_guard.get(&user_id) {
                        if user.is_active() {
                            // Simulate WebSocket delivery
                            Self::simulate_websocket_delivery(user_id, message);
                            delivery_count += 1;
                        }
                    }
                }
            }
        }
        
        delivery_count > 0 || room_users.len() <= 1 // Success if delivered to someone or only sender in room
    }

    fn simulate_websocket_delivery(user_id: u64, message: &Message) {
        // Simulate network latency for WebSocket delivery
        let latency_ms = fastrand::u64(1..50);
        std::thread::sleep(Duration::from_millis(latency_ms));
        
        // Simulate occasional delivery failures
        if fastrand::f64() < 0.02 { // 2% failure rate
            return;
        }
        
        // In a real implementation, this would send the message via WebSocket
        if fastrand::f64() < 0.1 { // 10% verbose logging
            println!("  → Delivered message {} to user {}", message.id, user_id);
        }
    }

    fn start_background_services(&self) -> Result<(), String> {
        // Heartbeat service
        let presence_tracker = self.presence_tracker.clone();
        let users = self.users.clone();
        let is_running = self.is_running.clone();

        let handle = self.runtime.spawn_fn_with_priority(move || {
            while is_running.load(Ordering::Relaxed) {
                // Clean up offline users
                if let Ok(cleaned) = presence_tracker.cleanup_offline_users() {
                    if cleaned > 0 {
                        println!("Heartbeat service: Cleaned up {} offline users", cleaned);
                    }
                }
                
                // Update user statuses
                if let Ok(users_guard) = users.read() {
                    for user in users_guard.values() {
                        let is_online = presence_tracker.is_user_online(user.id);
                        user.is_online.store(is_online, Ordering::Relaxed);
                    }
                }
                
                std::thread::sleep(Duration::from_secs(30)); // Run every 30 seconds
            }
        }, Priority::Low);

        std::mem::drop(handle);
        Ok(())
    }

    fn connect_user(&self, username: String) -> Result<Arc<User>, String> {
        // Check user limit
        if self.active_connections.load(Ordering::Relaxed) >= self.max_users {
            return Err("Server at capacity".to_string());
        }

        let user_id = self.next_user_id.fetch_add(1, Ordering::Relaxed);
        let session_id = format!("session_{}", fastrand::u64(..));
        let user = Arc::new(User::new(user_id, username, session_id));

        // Add to users map
        {
            let mut users = self.users.write()
                .map_err(|_| "Failed to acquire users lock")?;
            users.insert(user_id, user.clone());
        }

        // Update presence
        self.presence_tracker.update_user_activity(user_id)?;

        // Update statistics
        self.total_connections.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_add(1, Ordering::Relaxed);

        println!("User connected: {} (ID: {})", user.username, user.id);
        Ok(user)
    }

    fn disconnect_user(&self, user_id: u64) -> Result<(), String> {
        // Remove from all rooms
        if let Ok(rooms_guard) = self.rooms.read() {
            for room in rooms_guard.values() {
                let _ = room.remove_user(user_id);
            }
        }

        // Remove from users map
        let was_present = {
            let mut users = self.users.write()
                .map_err(|_| "Failed to acquire users lock")?;
            users.remove(&user_id).is_some()
        };

        if was_present {
            self.active_connections.fetch_sub(1, Ordering::Relaxed);
            println!("User disconnected: {}", user_id);
        }

        Ok(())
    }

    fn create_room(&self, room_name: String, topic: Option<String>) -> Result<Arc<ChatRoom>, String> {
        // Check room limit
        if self.rooms.read().unwrap().len() >= self.max_rooms {
            return Err("Maximum number of rooms reached".to_string());
        }

        let room = Arc::new(ChatRoom::new(room_name.clone(), topic));
        
        {
            let mut rooms = self.rooms.write()
                .map_err(|_| "Failed to acquire rooms lock")?;
            
            if rooms.contains_key(&room_name) {
                return Err("Room already exists".to_string());
            }
            
            rooms.insert(room_name.clone(), room.clone());
        }

        println!("Room created: {}", room_name);
        Ok(room)
    }

    fn join_room(&self, user_id: u64, room_name: String) -> Result<(), String> {
        // Get user
        let user = {
            let users = self.users.read()
                .map_err(|_| "Failed to acquire users lock")?;
            users.get(&user_id).cloned()
                .ok_or_else(|| "User not found".to_string())?
        };

        // Get room
        let room = {
            let rooms = self.rooms.read()
                .map_err(|_| "Failed to acquire rooms lock")?;
            rooms.get(&room_name).cloned()
                .ok_or_else(|| "Room not found".to_string())?
        };

        // Add user to room
        room.add_user(user_id)?;
        user.join_room(&room_name)?;

        // Send join message
        let join_message = Message::new(
            0, // System user
            "System".to_string(),
            room_name,
            MessageType::System,
            format!("{} joined the room", user.username),
        );

        self.send_message(join_message)?;
        println!("User {} joined room {}", user.username, room.name);
        Ok(())
    }

    fn leave_room(&self, user_id: u64, room_name: String) -> Result<(), String> {
        // Get user
        let user = {
            let users = self.users.read()
                .map_err(|_| "Failed to acquire users lock")?;
            users.get(&user_id).cloned()
                .ok_or_else(|| "User not found".to_string())?
        };

        // Get room
        let room = {
            let rooms = self.rooms.read()
                .map_err(|_| "Failed to acquire rooms lock")?;
            rooms.get(&room_name).cloned()
                .ok_or_else(|| "Room not found".to_string())?
        };

        // Remove user from room
        room.remove_user(user_id)?;
        user.leave_room(&room_name)?;

        // Send leave message
        let leave_message = Message::new(
            0, // System user
            "System".to_string(),
            room_name,
            MessageType::System,
            format!("{} left the room", user.username),
        );

        self.send_message(leave_message)?;
        println!("User {} left room {}", user.username, room.name);
        Ok(())
    }

    fn send_message(&self, message: Message) -> Result<(), String> {
        // Update user activity
        if message.sender_id != 0 { // Not a system message
            self.presence_tracker.update_user_activity(message.sender_id)?;
            
            // Increment user message count
            if let Ok(users) = self.users.read() {
                if let Some(user) = users.get(&message.sender_id) {
                    user.increment_message_count();
                }
            }
        }

        // Queue message for delivery
        self.message_queue.enqueue(message)?;
        self.messages_received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn send_typing_indicator(&self, user_id: u64, room_name: String, is_typing: bool) -> Result<(), String> {
        // Get room and update typing indicator
        if let Ok(rooms) = self.rooms.read() {
            if let Some(room) = rooms.get(&room_name) {
                room.update_typing_indicator(user_id, is_typing)?;
            }
        }

        // Send typing indicator message
        if let Ok(users) = self.users.read() {
            if let Some(user) = users.get(&user_id) {
                let typing_message = Message::new(
                    user_id,
                    user.username.clone(),
                    room_name,
                    MessageType::TypingIndicator,
                    if is_typing { "typing" } else { "stopped_typing" }.to_string(),
                );

                self.message_queue.enqueue(typing_message)?;
            }
        }

        Ok(())
    }

    fn get_room_info(&self, room_name: &str) -> Result<RoomInfo, String> {
        let rooms = self.rooms.read()
            .map_err(|_| "Failed to acquire rooms lock")?;
        
        let room = rooms.get(room_name)
            .ok_or_else(|| "Room not found".to_string())?;

        let recent_messages = room.get_recent_messages(50)?;
        let typing_users = room.get_typing_users();
        let user_list = room.get_user_list();

        Ok(RoomInfo {
            name: room.name.clone(),
            topic: room.topic.clone(),
            user_count: room.user_count(),
            message_count: room.message_count(),
            recent_messages,
            typing_users,
            user_list,
        })
    }

    fn get_server_stats(&self) -> ServerStats {
        let (queued, pending, sent, failed, avg_latency) = self.message_queue.stats();
        let online_users = self.presence_tracker.get_online_users();
        let uptime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - self.server_start_time;

        ServerStats {
            total_connections: self.total_connections.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            online_users: online_users.len(),
            total_rooms: self.rooms.read().unwrap().len(),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_queued: queued,
            messages_pending: pending,
            messages_failed: failed,
            avg_delivery_latency_ms: avg_latency,
            uptime_seconds: uptime,
        }
    }

    fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        println!("Chat server stopped");
    }
}

#[derive(Debug)]
struct RoomInfo {
    name: String,
    topic: Option<String>,
    user_count: usize,
    message_count: usize,
    recent_messages: Vec<Message>,
    typing_users: Vec<u64>,
    user_list: Vec<u64>,
}

#[derive(Debug)]
struct ServerStats {
    total_connections: usize,
    active_connections: usize,
    online_users: usize,
    total_rooms: usize,
    messages_received: usize,
    messages_sent: usize,
    messages_queued: usize,
    messages_pending: usize,
    messages_failed: usize,
    avg_delivery_latency_ms: u64,
    uptime_seconds: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real-time Chat Server - WebSocket Pub/Sub Simulation");
    println!("====================================================");

    // Create chat server
    let server = ChatServer::new(
        1000, // max users
        100,  // max rooms
        4,    // message workers
    )?;

    server.start()?;

    // Create some rooms
    println!("\n1. Creating chat rooms...");
    server.create_room("general".to_string(), Some("General discussion".to_string()))?;
    server.create_room("tech".to_string(), Some("Technology discussions".to_string()))?;
    server.create_room("random".to_string(), None)?;
    println!("  Created 3 chat rooms");

    // Simulate users connecting
    println!("\n2. Simulating user connections...");
    let mut users = Vec::new();
    for i in 0..20 {
        let username = format!("user_{}", i);
        let user = server.connect_user(username)?;
        users.push(user);
    }
    println!("  Connected {} users", users.len());

    // Users join rooms
    println!("\n3. Users joining rooms...");
    for (i, user) in users.iter().enumerate() {
        match i % 3 {
            0 => server.join_room(user.id, "general".to_string())?,
            1 => server.join_room(user.id, "tech".to_string())?,
            2 => server.join_room(user.id, "random".to_string())?,
            _ => {}
        }
        
        // Some users join multiple rooms
        if i % 5 == 0 {
            server.join_room(user.id, "general".to_string())?;
        }
    }

    // Simulate message activity
    println!("\n4. Simulating message activity...");
    let message_start = Instant::now();
    
    // Send various types of messages
    for round in 0..5 {
        println!("  Round {}: Sending messages...", round + 1);
        
        for (i, user) in users.iter().enumerate() {
            let room_name = match i % 3 {
                0 => "general",
                1 => "tech", 
                2 => "random",
                _ => "general",
            };

            // Send typing indicator
            server.send_typing_indicator(user.id, room_name.to_string(), true)?;
            
            // Simulate typing delay
            std::thread::sleep(Duration::from_millis(fastrand::u64(100..500)));
            
            // Send message
            let message_content = match round {
                0 => format!("Hello everyone! This is {} joining the conversation.", user.username),
                1 => format!("I'm working on some interesting {} project!", if i % 2 == 0 { "Rust" } else { "AI" }),
                2 => format!("Does anyone know about {}?", if i % 2 == 0 { "async programming" } else { "concurrency patterns" }),
                3 => format!("Great discussion! I learned a lot about {}.", if i % 2 == 0 { "Moirai" } else { "real-time systems" }),
                4 => format!("Thanks for the chat, everyone! See you later from {}.", user.username),
                _ => format!("Message {} from {}", round, user.username),
            };

            let message = Message::new(
                user.id,
                user.username.clone(),
                room_name.to_string(),
                MessageType::Text,
                message_content,
            );

            server.send_message(message)?;
            
            // Stop typing
            server.send_typing_indicator(user.id, room_name.to_string(), false)?;
            
            // Random delay between messages
            if i % 3 == 0 {
                std::thread::sleep(Duration::from_millis(fastrand::u64(50..200)));
            }
        }
        
        // Wait between rounds
        std::thread::sleep(Duration::from_millis(500));
    }

    let message_time = message_start.elapsed();
    println!("  Message activity completed in {:?}", message_time);

    // Simulate some reactions and file sharing
    println!("\n5. Simulating reactions and file sharing...");
    for user in users.iter().take(5) {
        // Send reaction
        let reaction_message = Message::new(
            user.id,
            user.username.clone(),
            "general".to_string(),
            MessageType::Reaction,
            "👍".to_string(),
        );
        server.send_message(reaction_message)?;

        // Send file sharing message
        let file_message = Message::new(
            user.id,
            user.username.clone(),
            "tech".to_string(),
            MessageType::File,
            format!("shared: project_report_{}.pdf", user.id),
        );
        server.send_message(file_message)?;
    }

    // Wait for message processing
    std::thread::sleep(Duration::from_secs(2));

    // Display room information
    println!("\n6. Room Information:");
    for room_name in ["general", "tech", "random"] {
        match server.get_room_info(room_name) {
            Ok(info) => {
                println!("  Room: {}", info.name);
                println!("    ├─ Topic: {}", info.topic.unwrap_or("None".to_string()));
                println!("    ├─ Users: {}", info.user_count);
                println!("    ├─ Messages: {}", info.message_count);
                println!("    ├─ Typing: {}", info.typing_users.len());
                println!("    └─ Recent messages: {}", info.recent_messages.len());
            }
            Err(e) => println!("  Failed to get info for {}: {}", room_name, e),
        }
    }

    // Simulate some users leaving
    println!("\n7. Simulating user departures...");
    for user in users.iter().take(5) {
        server.leave_room(user.id, "general".to_string())?;
        server.disconnect_user(user.id)?;
    }

    // Wait for cleanup
    std::thread::sleep(Duration::from_millis(500));

    // Display comprehensive server statistics
    println!("\n8. Final Server Statistics:");
    let stats = server.get_server_stats();
    
    println!("  ├─ Connections:");
    println!("  │  ├─ Total connections: {}", stats.total_connections);
    println!("  │  ├─ Active connections: {}", stats.active_connections);
    println!("  │  └─ Online users: {}", stats.online_users);
    
    println!("  ├─ Rooms:");
    println!("  │  └─ Total rooms: {}", stats.total_rooms);
    
    println!("  ├─ Messages:");
    println!("  │  ├─ Received: {}", stats.messages_received);
    println!("  │  ├─ Sent: {}", stats.messages_sent);
    println!("  │  ├─ Queued: {}", stats.messages_queued);
    println!("  │  ├─ Pending: {}", stats.messages_pending);
    println!("  │  ├─ Failed: {}", stats.messages_failed);
    println!("  │  └─ Success rate: {:.1}%", 
             (stats.messages_sent as f64 / stats.messages_received.max(1) as f64) * 100.0);
    
    println!("  ├─ Performance:");
    println!("  │  ├─ Avg delivery latency: {}ms", stats.avg_delivery_latency_ms);
    println!("  │  ├─ Message throughput: {:.1} msg/sec", 
             stats.messages_sent as f64 / message_time.as_secs_f64());
    println!("  │  └─ Concurrent efficiency: {:.1}%", 
             (stats.messages_sent as f64 / (4.0 * message_time.as_secs_f64())) * 100.0);
    
    println!("  └─ Uptime: {}m {}s", stats.uptime_seconds / 60, stats.uptime_seconds % 60);

    // Stop server
    server.stop();

    println!("\nReal-time chat server demonstration completed!");
    println!("Successfully demonstrated:");
    println!("- WebSocket-style real-time message handling");
    println!("- Publish/subscribe system with room-based routing");
    println!("- Concurrent user session management");
    println!("- Message queuing with priority and delivery guarantees");
    println!("- Real-time presence tracking and typing indicators");
    println!("- Event-driven architecture with message broadcasting");

    Ok(())
}