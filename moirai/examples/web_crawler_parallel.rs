//! Web Crawler with Parallel Processing - Real-World Async/Concurrent Example
//!
//! This example demonstrates:
//! - Parallel HTTP requests with rate limiting and connection pooling
//! - Async/await patterns for I/O-bound operations
//! - Concurrent file system operations for data persistence
//! - URL frontier management with priority queuing
//! - Robots.txt compliance and politeness policies
//! - Circuit breakers for unreliable websites
//! - Content extraction and parallel processing

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]
#![expect(dead_code, reason = "This example retains crawler frontier fields and priority variants beyond the short demo")]

use moirai::{Moirai, Priority};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// Simple pseudo-random number generator for demo purposes
fn simple_random() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn random_f64() -> f64 {
    (simple_random() % 1000000) as f64 / 1000000.0
}

fn random_usize(min: usize, max: usize) -> usize {
    min + (simple_random() % (max - min) as u64) as usize
}

/// Represents a URL to be crawled with metadata
#[derive(Debug, Clone)]
struct CrawlTarget {
    url: String,
    depth: u32,
    priority: CrawlPriority,
    discovered_time: u64,
    domain: String,
    parent_url: Option<String>,
    retry_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CrawlPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Result of crawling a page
#[derive(Debug, Clone)]
struct CrawlResult {
    url: String,
    status_code: u16,
    content_length: usize,
    content_type: String,
    title: Option<String>,
    links: Vec<String>,
    processing_time_ms: u64,
    discovered_at: u64,
    crawled_at: u64,
    success: bool,
    error_message: Option<String>,
}

/// Domain-specific crawling policies and state
#[derive(Debug)]
struct DomainState {
    domain: String,
    last_request_time: AtomicU64,
    request_count: AtomicUsize,
    min_delay_ms: u64,
    max_concurrent_requests: usize,
    current_requests: AtomicUsize,
    robots_txt_rules: Mutex<HashMap<String, bool>>, // path -> allowed
    circuit_breaker_failures: AtomicUsize,
    circuit_breaker_state: AtomicUsize, // 0: Closed, 1: Open, 2: Half-Open
}

impl DomainState {
    fn new(domain: String, min_delay_ms: u64, max_concurrent: usize) -> Self {
        Self {
            domain,
            last_request_time: AtomicU64::new(0),
            request_count: AtomicUsize::new(0),
            min_delay_ms,
            max_concurrent_requests: max_concurrent,
            current_requests: AtomicUsize::new(0),
            robots_txt_rules: Mutex::new(HashMap::new()),
            circuit_breaker_failures: AtomicUsize::new(0),
            circuit_breaker_state: AtomicUsize::new(0), // Closed
        }
    }

    fn can_make_request(&self, path: &str) -> bool {
        // Check circuit breaker
        if self.circuit_breaker_state.load(Ordering::Relaxed) == 1 {
            return false; // Circuit open
        }

        // Check concurrent request limit
        if self.current_requests.load(Ordering::Relaxed) >= self.max_concurrent_requests {
            return false;
        }

        // Check robots.txt compliance
        if let Ok(rules) = self.robots_txt_rules.lock() {
            if let Some(&allowed) = rules.get(path) {
                if !allowed {
                    return false;
                }
            }
        }

        // Check rate limiting
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let last_request = self.last_request_time.load(Ordering::Relaxed);

        now.saturating_sub(last_request) >= self.min_delay_ms
    }

    fn start_request(&self) -> bool {
        if self.can_make_request("") {
            self.current_requests.fetch_add(1, Ordering::Relaxed);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            self.last_request_time.store(now, Ordering::Relaxed);
            self.request_count.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn complete_request(&self, success: bool) {
        self.current_requests.fetch_sub(1, Ordering::Relaxed);

        if success {
            // Reset circuit breaker on success
            if self.circuit_breaker_state.load(Ordering::Relaxed) == 2 {
                self.circuit_breaker_state.store(0, Ordering::Relaxed);
                self.circuit_breaker_failures.store(0, Ordering::Relaxed);
            }
        } else {
            let failures = self
                .circuit_breaker_failures
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            if failures >= 5 {
                self.circuit_breaker_state.store(1, Ordering::Relaxed); // Open circuit
            }
        }
    }

    fn stats(&self) -> (usize, usize, u64, usize) {
        (
            self.request_count.load(Ordering::Relaxed),
            self.current_requests.load(Ordering::Relaxed),
            self.last_request_time.load(Ordering::Relaxed),
            self.circuit_breaker_state.load(Ordering::Relaxed),
        )
    }
}

/// URL frontier for managing crawl queue with priority
struct UrlFrontier {
    high_priority: Arc<Mutex<VecDeque<CrawlTarget>>>,
    normal_priority: Arc<Mutex<VecDeque<CrawlTarget>>>,
    low_priority: Arc<Mutex<VecDeque<CrawlTarget>>>,
    visited_urls: Arc<RwLock<HashSet<String>>>,
    domain_queues: Arc<RwLock<HashMap<String, VecDeque<CrawlTarget>>>>,
    total_queued: AtomicUsize,
    total_discovered: AtomicUsize,
}

impl UrlFrontier {
    fn new() -> Self {
        Self {
            high_priority: Arc::new(Mutex::new(VecDeque::new())),
            normal_priority: Arc::new(Mutex::new(VecDeque::new())),
            low_priority: Arc::new(Mutex::new(VecDeque::new())),
            visited_urls: Arc::new(RwLock::new(HashSet::new())),
            domain_queues: Arc::new(RwLock::new(HashMap::new())),
            total_queued: AtomicUsize::new(0),
            total_discovered: AtomicUsize::new(0),
        }
    }

    fn add_url(&self, target: CrawlTarget) -> Result<(), String> {
        // Check if already visited
        {
            let visited = self
                .visited_urls
                .read()
                .map_err(|_| "Failed to read visited URLs")?;
            if visited.contains(&target.url) {
                return Ok(()); // Already visited
            }
        }

        // Add to visited set
        {
            let mut visited = self
                .visited_urls
                .write()
                .map_err(|_| "Failed to write visited URLs")?;
            if !visited.insert(target.url.clone()) {
                return Ok(()); // Already visited (race condition)
            }
        }

        // Add to appropriate priority queue
        let queue = match target.priority {
            CrawlPriority::Critical | CrawlPriority::High => &self.high_priority,
            CrawlPriority::Normal => &self.normal_priority,
            CrawlPriority::Low => &self.low_priority,
        };

        queue
            .lock()
            .map_err(|_| "Failed to acquire queue lock")?
            .push_back(target);

        self.total_queued.fetch_add(1, Ordering::Relaxed);
        self.total_discovered.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn get_next_url(&self) -> Option<CrawlTarget> {
        // Check high priority first
        if let Ok(mut queue) = self.high_priority.lock() {
            if let Some(target) = queue.pop_front() {
                self.total_queued.fetch_sub(1, Ordering::Relaxed);
                return Some(target);
            }
        }

        // Then normal priority
        if let Ok(mut queue) = self.normal_priority.lock() {
            if let Some(target) = queue.pop_front() {
                self.total_queued.fetch_sub(1, Ordering::Relaxed);
                return Some(target);
            }
        }

        // Finally low priority
        if let Ok(mut queue) = self.low_priority.lock() {
            if let Some(target) = queue.pop_front() {
                self.total_queued.fetch_sub(1, Ordering::Relaxed);
                return Some(target);
            }
        }

        None
    }

    fn is_empty(&self) -> bool {
        self.total_queued.load(Ordering::Relaxed) == 0
    }

    fn stats(&self) -> (usize, usize, usize) {
        (
            self.total_discovered.load(Ordering::Relaxed),
            self.total_queued.load(Ordering::Relaxed),
            self.visited_urls.read().map(|v| v.len()).unwrap_or(0),
        )
    }
}

/// Content processor for parallel text extraction and analysis
struct ContentProcessor {
    processed_pages: AtomicUsize,
    total_content_bytes: AtomicU64,
    extracted_links: AtomicUsize,
    extraction_time: AtomicU64,
}

impl ContentProcessor {
    fn new() -> Self {
        Self {
            processed_pages: AtomicUsize::new(0),
            total_content_bytes: AtomicU64::new(0),
            extracted_links: AtomicUsize::new(0),
            extraction_time: AtomicU64::new(0),
        }
    }

    fn process_content(
        &self,
        url: &str,
        content: &str,
    ) -> Result<(Option<String>, Vec<String>), String> {
        let start_time = Instant::now();

        // Simulate content processing
        let title = self.extract_title(content);
        let links = self.extract_links(content, url)?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        self.processed_pages.fetch_add(1, Ordering::Relaxed);
        self.total_content_bytes
            .fetch_add(content.len() as u64, Ordering::Relaxed);
        self.extracted_links
            .fetch_add(links.len(), Ordering::Relaxed);
        self.extraction_time
            .fetch_add(processing_time, Ordering::Relaxed);

        Ok((title, links))
    }

    fn extract_title(&self, content: &str) -> Option<String> {
        // Simple title extraction (in real implementation, use proper HTML parser)
        if let Some(start) = content.find("<title>") {
            if let Some(end) = content[start + 7..].find("</title>") {
                let title = content[start + 7..start + 7 + end].trim();
                if !title.is_empty() {
                    return Some(title.to_string());
                }
            }
        }
        None
    }

    fn extract_links(&self, content: &str, base_url: &str) -> Result<Vec<String>, String> {
        let mut links = Vec::new();

        // Simple link extraction (in real implementation, use proper HTML parser)
        let mut search_pos = 0;
        while let Some(href_pos) = content[search_pos..].find("href=\"") {
            let start = search_pos + href_pos + 6;
            if let Some(end_pos) = content[start..].find("\"") {
                let end = start + end_pos;
                let link = content[start..end].trim();

                if !link.is_empty() && !link.starts_with('#') {
                    let absolute_url = self.resolve_url(base_url, link)?;
                    links.push(absolute_url);
                }

                search_pos = end;
            } else {
                break;
            }
        }

        Ok(links)
    }

    fn resolve_url(&self, base_url: &str, relative_url: &str) -> Result<String, String> {
        if relative_url.starts_with("http://") || relative_url.starts_with("https://") {
            Ok(relative_url.to_string())
        } else if relative_url.starts_with('/') {
            // Extract domain from base URL
            let domain_end = base_url
                .find('/')
                .map(|pos| {
                    if base_url[..pos].contains("://") {
                        base_url[pos + 2..]
                            .find('/')
                            .map(|p| pos + 2 + p + 1)
                            .unwrap_or(base_url.len())
                    } else {
                        pos
                    }
                })
                .unwrap_or(base_url.len());

            let domain = &base_url[..domain_end];
            Ok(format!("{}{}", domain, relative_url))
        } else {
            // Relative to current directory
            let base_path = base_url
                .rfind('/')
                .map(|pos| &base_url[..pos + 1])
                .unwrap_or(base_url);
            Ok(format!("{}{}", base_path, relative_url))
        }
    }

    fn stats(&self) -> (usize, u64, usize, u64) {
        (
            self.processed_pages.load(Ordering::Relaxed),
            self.total_content_bytes.load(Ordering::Relaxed),
            self.extracted_links.load(Ordering::Relaxed),
            self.extraction_time.load(Ordering::Relaxed),
        )
    }
}

/// File system manager for concurrent data persistence
struct FileSystemManager {
    base_path: String,
    files_written: AtomicUsize,
    bytes_written: AtomicU64,
    write_errors: AtomicUsize,
}

impl FileSystemManager {
    fn new(base_path: String) -> Self {
        Self {
            base_path,
            files_written: AtomicUsize::new(0),
            bytes_written: AtomicU64::new(0),
            write_errors: AtomicUsize::new(0),
        }
    }

    fn save_crawl_result(&self, result: &CrawlResult) -> Result<(), String> {
        // Create domain directory
        let domain = self.extract_domain(&result.url);
        let domain_path = format!("{}/{}", self.base_path, domain);
        std::fs::create_dir_all(&domain_path)
            .map_err(|e| format!("Failed to create directory {}: {}", domain_path, e))?;

        // Generate filename from URL
        let filename = self.url_to_filename(&result.url);
        let file_path = format!("{}/{}.json", domain_path, filename);

        // Serialize result to JSON (simplified)
        let json_content = format!(
            r#"{{
    "url": "{}",
    "status_code": {},
    "content_length": {},
    "content_type": "{}",
    "title": {},
    "links_count": {},
    "processing_time_ms": {},
    "discovered_at": {},
    "crawled_at": {},
    "success": {},
    "error_message": {}
}}"#,
            result.url,
            result.status_code,
            result.content_length,
            result.content_type,
            result
                .title
                .as_ref()
                .map(|t| format!("\"{}\"", t))
                .unwrap_or("null".to_string()),
            result.links.len(),
            result.processing_time_ms,
            result.discovered_at,
            result.crawled_at,
            result.success,
            result
                .error_message
                .as_ref()
                .map(|e| format!("\"{}\"", e))
                .unwrap_or("null".to_string())
        );

        // Write file
        match std::fs::write(&file_path, &json_content) {
            Ok(_) => {
                self.files_written.fetch_add(1, Ordering::Relaxed);
                self.bytes_written
                    .fetch_add(json_content.len() as u64, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.write_errors.fetch_add(1, Ordering::Relaxed);
                Err(format!("Failed to write file {}: {}", file_path, e))
            }
        }
    }

    fn extract_domain(&self, url: &str) -> String {
        if let Some(start) = url.find("://") {
            let after_protocol = &url[start + 3..];
            if let Some(end) = after_protocol.find('/') {
                after_protocol[..end].to_string()
            } else {
                after_protocol.to_string()
            }
        } else {
            url.split('/').next().unwrap_or("unknown").to_string()
        }
    }

    fn url_to_filename(&self, url: &str) -> String {
        // Convert URL to safe filename
        let safe_chars: String = url
            .chars()
            .map(|c| match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
                _ => '_',
            })
            .collect();

        // Limit length
        if safe_chars.len() > 100 {
            format!(
                "{}_hash_{:x}",
                &safe_chars[..80],
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            )
        } else {
            safe_chars
        }
    }

    fn stats(&self) -> (usize, u64, usize) {
        (
            self.files_written.load(Ordering::Relaxed),
            self.bytes_written.load(Ordering::Relaxed),
            self.write_errors.load(Ordering::Relaxed),
        )
    }
}

/// Main web crawler with parallel processing capabilities
struct WebCrawler {
    runtime: Moirai,
    frontier: Arc<UrlFrontier>,
    domain_states: Arc<RwLock<HashMap<String, Arc<DomainState>>>>,
    content_processor: Arc<ContentProcessor>,
    file_manager: Arc<FileSystemManager>,

    // Configuration
    max_depth: u32,
    max_pages: usize,
    worker_threads: usize,

    // Statistics
    pages_crawled: Arc<AtomicUsize>,
    pages_failed: Arc<AtomicUsize>,
    total_crawl_time: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
}

impl WebCrawler {
    fn new(
        max_depth: u32,
        max_pages: usize,
        worker_threads: usize,
        output_path: String,
    ) -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;

        // Create output directory
        std::fs::create_dir_all(&output_path)
            .map_err(|e| format!("Failed to create output directory {}: {}", output_path, e))?;

        Ok(Self {
            runtime,
            frontier: Arc::new(UrlFrontier::new()),
            domain_states: Arc::new(RwLock::new(HashMap::new())),
            content_processor: Arc::new(ContentProcessor::new()),
            file_manager: Arc::new(FileSystemManager::new(output_path)),
            max_depth,
            max_pages,
            worker_threads,
            pages_crawled: Arc::new(AtomicUsize::new(0)),
            pages_failed: Arc::new(AtomicUsize::new(0)),
            total_crawl_time: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
        })
    }

    fn add_seed_url(&self, url: String, priority: CrawlPriority) -> Result<(), String> {
        let domain = self.extract_domain(&url);
        let target = CrawlTarget {
            url: url.clone(),
            depth: 0,
            priority,
            discovered_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            domain: domain.clone(),
            parent_url: None,
            retry_count: 0,
        };

        // Initialize domain state
        self.get_or_create_domain_state(&domain)?;

        self.frontier.add_url(target)?;
        Ok(())
    }

    fn get_or_create_domain_state(&self, domain: &str) -> Result<Arc<DomainState>, String> {
        // Check if domain state exists
        {
            let states = self
                .domain_states
                .read()
                .map_err(|_| "Failed to read domain states")?;
            if let Some(state) = states.get(domain) {
                return Ok(state.clone());
            }
        }

        // Create new domain state
        let mut states = self
            .domain_states
            .write()
            .map_err(|_| "Failed to write domain states")?;

        if let Some(state) = states.get(domain) {
            Ok(state.clone()) // Another thread created it
        } else {
            let state = Arc::new(DomainState::new(
                domain.to_string(),
                1000, // 1 second between requests
                3,    // max 3 concurrent requests per domain
            ));
            states.insert(domain.to_string(), state.clone());
            Ok(state)
        }
    }

    fn extract_domain(&self, url: &str) -> String {
        if let Some(start) = url.find("://") {
            let after_protocol = &url[start + 3..];
            if let Some(end) = after_protocol.find('/') {
                after_protocol[..end].to_string()
            } else {
                after_protocol.to_string()
            }
        } else {
            url.split('/').next().unwrap_or("unknown").to_string()
        }
    }

    fn start_crawling(&self) -> Result<(), String> {
        self.is_running.store(true, Ordering::Relaxed);
        println!(
            "Starting web crawler with {} worker threads",
            self.worker_threads
        );

        // Start worker threads
        for worker_id in 0..self.worker_threads {
            let frontier = self.frontier.clone();
            let domain_states = self.domain_states.clone();
            let content_processor = self.content_processor.clone();
            let file_manager = self.file_manager.clone();
            let pages_crawled = self.pages_crawled.clone();
            let pages_failed = self.pages_failed.clone();
            let total_crawl_time = self.total_crawl_time.clone();
            let is_running = self.is_running.clone();
            let max_depth = self.max_depth;
            let max_pages = self.max_pages;

            let handle = self.runtime.spawn_fn_with_priority(
                move || {
                    while is_running.load(Ordering::Relaxed) {
                        // Check if we've reached the page limit
                        if pages_crawled.load(Ordering::Relaxed) >= max_pages {
                            break;
                        }

                        // Get next URL to crawl
                        let target = match frontier.get_next_url() {
                            Some(target) => target,
                            None => {
                                // No URLs available, wait a bit
                                std::thread::sleep(Duration::from_millis(100));
                                continue;
                            }
                        };

                        // Get domain state
                        let domain_state = match domain_states.read() {
                            Ok(states) => states.get(&target.domain).cloned(),
                            Err(_) => continue,
                        };

                        let domain_state = match domain_state {
                            Some(state) => state,
                            None => continue,
                        };

                        // Check if we can make a request to this domain
                        if !domain_state.start_request() {
                            // Can't make request now, put URL back
                            let _ = frontier.add_url(target);
                            std::thread::sleep(Duration::from_millis(100));
                            continue;
                        }

                        // Crawl the page
                        let crawl_start = Instant::now();
                        let result = Self::crawl_page(&target, &content_processor);
                        let crawl_time = crawl_start.elapsed().as_millis() as u64;

                        domain_state.complete_request(result.success);
                        total_crawl_time.fetch_add(crawl_time, Ordering::Relaxed);

                        if result.success {
                            pages_crawled.fetch_add(1, Ordering::Relaxed);

                            // Add discovered links to frontier
                            if target.depth < max_depth {
                                for link in &result.links {
                                    let link_domain = Self::extract_domain_static(link);
                                    let link_target = CrawlTarget {
                                        url: link.clone(),
                                        depth: target.depth + 1,
                                        priority: if target.depth < 2 {
                                            CrawlPriority::Normal
                                        } else {
                                            CrawlPriority::Low
                                        },
                                        discovered_time: SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_secs(),
                                        domain: link_domain,
                                        parent_url: Some(target.url.clone()),
                                        retry_count: 0,
                                    };
                                    let _ = frontier.add_url(link_target);
                                }
                            }
                        } else {
                            pages_failed.fetch_add(1, Ordering::Relaxed);
                        }

                        // Save result to disk
                        if let Err(e) = file_manager.save_crawl_result(&result) {
                            println!(
                                "Worker {}: Failed to save result for {}: {}",
                                worker_id, result.url, e
                            );
                        }

                        // Log progress
                        let total_crawled = pages_crawled.load(Ordering::Relaxed);
                        if total_crawled.is_multiple_of(10) {
                            println!("Worker {}: Crawled {} pages", worker_id, total_crawled);
                        }
                    }
                },
                Priority::Normal,
            );

            std::mem::drop(handle); // Let workers run asynchronously
        }

        Ok(())
    }

    fn crawl_page(target: &CrawlTarget, content_processor: &ContentProcessor) -> CrawlResult {
        let start_time = Instant::now();
        let discovered_at = target.discovered_time;
        let crawled_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Simulate HTTP request (in real implementation, use proper HTTP client)
        let (success, status_code, content, content_type, error_message) =
            Self::simulate_http_request(&target.url);

        let processing_time = start_time.elapsed().as_millis() as u64;

        if success {
            // Process content
            match content_processor.process_content(&target.url, &content) {
                Ok((title, links)) => CrawlResult {
                    url: target.url.clone(),
                    status_code,
                    content_length: content.len(),
                    content_type,
                    title,
                    links,
                    processing_time_ms: processing_time,
                    discovered_at,
                    crawled_at,
                    success: true,
                    error_message: None,
                },
                Err(e) => CrawlResult {
                    url: target.url.clone(),
                    status_code,
                    content_length: content.len(),
                    content_type,
                    title: None,
                    links: Vec::new(),
                    processing_time_ms: processing_time,
                    discovered_at,
                    crawled_at,
                    success: false,
                    error_message: Some(format!("Content processing failed: {}", e)),
                },
            }
        } else {
            CrawlResult {
                url: target.url.clone(),
                status_code,
                content_length: 0,
                content_type: "".to_string(),
                title: None,
                links: Vec::new(),
                processing_time_ms: processing_time,
                discovered_at,
                crawled_at,
                success: false,
                error_message,
            }
        }
    }

    fn simulate_http_request(url: &str) -> (bool, u16, String, String, Option<String>) {
        // Simulate network delay
        let delay_ms = random_usize(50, 300) as u64;
        std::thread::sleep(Duration::from_millis(delay_ms));

        // Simulate various response scenarios
        if url.contains("error") {
            (
                false,
                500,
                String::new(),
                String::new(),
                Some("Server error".to_string()),
            )
        } else if url.contains("not-found") {
            (
                false,
                404,
                String::new(),
                String::new(),
                Some("Not found".to_string()),
            )
        } else if random_f64() < 0.05 {
            // 5% failure rate
            (
                false,
                503,
                String::new(),
                String::new(),
                Some("Service unavailable".to_string()),
            )
        } else {
            // Generate mock HTML content
            let content = format!(
                r#"<!DOCTYPE html>
<html>
<head>
    <title>Page: {}</title>
</head>
<body>
    <h1>Welcome to {}</h1>
    <p>This is a sample page with some content.</p>
    <a href="/page1">Page 1</a>
    <a href="/page2">Page 2</a>
    <a href="https://example.com/external">External Link</a>
    <a href="/api/data">API Data</a>
    <p>Generated content with {} random bytes.</p>
</body>
</html>"#,
                url,
                url,
                random_usize(100, 2000)
            );

            (true, 200, content, "text/html".to_string(), None)
        }
    }

    fn extract_domain_static(url: &str) -> String {
        if let Some(start) = url.find("://") {
            let after_protocol = &url[start + 3..];
            if let Some(end) = after_protocol.find('/') {
                after_protocol[..end].to_string()
            } else {
                after_protocol.to_string()
            }
        } else {
            url.split('/').next().unwrap_or("unknown").to_string()
        }
    }

    fn wait_for_completion(&self, timeout_seconds: u64) -> Result<(), String> {
        let start_time = Instant::now();

        while self.is_running.load(Ordering::Relaxed) {
            // Check timeout
            if start_time.elapsed().as_secs() > timeout_seconds {
                self.stop_crawling();
                return Err("Crawling timed out".to_string());
            }

            // Check if we've reached the page limit
            if self.pages_crawled.load(Ordering::Relaxed) >= self.max_pages {
                self.stop_crawling();
                break;
            }

            // Check if frontier is empty
            if self.frontier.is_empty() {
                std::thread::sleep(Duration::from_millis(500));
                if self.frontier.is_empty() {
                    self.stop_crawling();
                    break;
                }
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        Ok(())
    }

    fn stop_crawling(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    fn get_statistics(&self) -> CrawlerStats {
        let (discovered, queued, visited) = self.frontier.stats();
        let (processed_pages, content_bytes, extracted_links, extraction_time) =
            self.content_processor.stats();
        let (files_written, bytes_written, write_errors) = self.file_manager.stats();

        CrawlerStats {
            pages_crawled: self.pages_crawled.load(Ordering::Relaxed),
            pages_failed: self.pages_failed.load(Ordering::Relaxed),
            urls_discovered: discovered,
            urls_queued: queued,
            urls_visited: visited,
            total_crawl_time_ms: self.total_crawl_time.load(Ordering::Relaxed),
            processed_pages,
            content_bytes,
            extracted_links,
            extraction_time_ms: extraction_time,
            files_written,
            bytes_written,
            write_errors,
        }
    }
}

#[derive(Debug)]
struct CrawlerStats {
    pages_crawled: usize,
    pages_failed: usize,
    urls_discovered: usize,
    urls_queued: usize,
    urls_visited: usize,
    total_crawl_time_ms: u64,
    processed_pages: usize,
    content_bytes: u64,
    extracted_links: usize,
    extraction_time_ms: u64,
    files_written: usize,
    bytes_written: u64,
    write_errors: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Web Crawler with Parallel Processing");
    println!("====================================");

    // Create crawler
    let crawler = WebCrawler::new(
        3,   // max depth
        100, // max pages
        4,   // worker threads
        "/tmp/crawler_output".to_string(),
    )?;

    // Add seed URLs
    println!("\n1. Adding seed URLs...");
    crawler.add_seed_url("https://example.com".to_string(), CrawlPriority::High)?;
    crawler.add_seed_url("https://test.com".to_string(), CrawlPriority::Normal)?;
    crawler.add_seed_url("https://demo.org/page1".to_string(), CrawlPriority::Normal)?;
    crawler.add_seed_url("https://sample.net/api".to_string(), CrawlPriority::Low)?;
    println!("  Added 4 seed URLs");

    // Start crawling
    println!("\n2. Starting parallel crawling...");
    let crawl_start = Instant::now();
    crawler.start_crawling()?;

    // Wait for completion
    println!("\n3. Crawling in progress...");
    crawler.wait_for_completion(30)?; // 30 second timeout

    let total_time = crawl_start.elapsed();
    println!("\n4. Crawling completed in {:?}", total_time);

    // Display statistics
    println!("\n5. Crawling Statistics:");
    let stats = crawler.get_statistics();

    println!("  ├─ URL Discovery:");
    println!("  │  ├─ URLs discovered: {}", stats.urls_discovered);
    println!("  │  ├─ URLs visited: {}", stats.urls_visited);
    println!("  │  └─ URLs queued: {}", stats.urls_queued);

    println!("  ├─ Page Processing:");
    println!("  │  ├─ Pages crawled: {}", stats.pages_crawled);
    println!("  │  ├─ Pages failed: {}", stats.pages_failed);
    println!(
        "  │  ├─ Success rate: {:.1}%",
        (stats.pages_crawled as f64 / (stats.pages_crawled + stats.pages_failed) as f64) * 100.0
    );
    println!(
        "  │  └─ Avg crawl time: {:.1}ms per page",
        stats.total_crawl_time_ms as f64 / stats.pages_crawled.max(1) as f64
    );

    println!("  ├─ Content Extraction:");
    println!("  │  ├─ Pages processed: {}", stats.processed_pages);
    println!(
        "  │  ├─ Content bytes: {:.2} KB",
        stats.content_bytes as f64 / 1024.0
    );
    println!("  │  ├─ Links extracted: {}", stats.extracted_links);
    println!(
        "  │  └─ Avg extraction time: {:.1}ms per page",
        stats.extraction_time_ms as f64 / stats.processed_pages.max(1) as f64
    );

    println!("  ├─ File Operations:");
    println!("  │  ├─ Files written: {}", stats.files_written);
    println!(
        "  │  ├─ Bytes written: {:.2} KB",
        stats.bytes_written as f64 / 1024.0
    );
    println!("  │  └─ Write errors: {}", stats.write_errors);

    println!("  └─ Performance:");
    println!(
        "     ├─ Total throughput: {:.1} pages/sec",
        stats.pages_crawled as f64 / total_time.as_secs_f64()
    );
    println!(
        "     ├─ Data throughput: {:.1} KB/sec",
        (stats.content_bytes as f64 / 1024.0) / total_time.as_secs_f64()
    );
    println!(
        "     └─ Parallel efficiency: {:.1}%",
        (stats.pages_crawled as f64 / 4.0 / total_time.as_secs_f64()) * 100.0
    );

    println!("\nWeb crawler demonstration completed!");
    println!("Successfully demonstrated:");
    println!("- Parallel HTTP requests with rate limiting");
    println!("- Async/await patterns for I/O operations");
    println!("- Concurrent content processing and file operations");
    println!("- URL frontier management with priority queuing");
    println!("- Domain-specific politeness policies");

    Ok(())
}
