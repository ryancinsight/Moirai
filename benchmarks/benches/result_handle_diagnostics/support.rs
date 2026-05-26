fn verify_ready_value(value: usize) -> usize {
    assert_eq!(value, READY_VALUE);
    black_box(value)
}

fn verify_captured_ready_value(value: usize) -> usize {
    assert_eq!(value, CAPTURED_READY_VALUE);
    black_box(value)
}

fn verify_max_inline_captured_value(value: usize) -> usize {
    assert_eq!(value, MAX_INLINE_CAPTURE_WORDS);
    black_box(value)
}

fn verify_oversized_captured_ready_value(value: usize) -> usize {
    assert_eq!(value, OVERSIZED_CAPTURED_READY_VALUE);
    black_box(value)
}

fn saturated_recorded_nanos(duration: Duration) -> usize {
    duration
        .as_nanos()
        .min((LIFECYCLE_TIMESTAMP_NOT_RECORDED - 1) as u128) as usize
}

fn elapsed_nanos_since(origin: Instant) -> usize {
    saturated_recorded_nanos(origin.elapsed())
}

fn duration_nanos_between(origin: Instant, later: Instant) -> usize {
    saturated_recorded_nanos(later.saturating_duration_since(origin))
}

#[cfg(windows)]
static QPC_TICKS_PER_SECOND: AtomicI64 = AtomicI64::new(0);

#[cfg(windows)]
fn query_performance_counter() -> i64 {
    let mut ticks = 0i64;
    // Safety: `ticks` is a valid out pointer for the Win32 API call.
    let ok = unsafe { QueryPerformanceCounter(&mut ticks) };
    assert_ne!(ok, 0, "QueryPerformanceCounter must succeed");
    ticks
}

#[cfg(windows)]
fn qpc_ticks_per_second() -> i64 {
    let cached = QPC_TICKS_PER_SECOND.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }

    let frequency = query_performance_frequency();
    QPC_TICKS_PER_SECOND.store(frequency, Ordering::Relaxed);
    frequency
}

#[cfg(windows)]
fn query_performance_frequency() -> i64 {
    let mut frequency = 0i64;
    // Safety: `frequency` is a valid out pointer for the Win32 API call.
    let ok = unsafe { QueryPerformanceFrequency(&mut frequency) };
    assert_ne!(ok, 0, "QueryPerformanceFrequency must succeed");
    frequency
}

#[cfg(windows)]
fn ticks_to_nanos(ticks: i64, ticks_per_second: i64) -> usize {
    let nanos = (ticks.max(0) as u128).saturating_mul(1_000_000_000) / (ticks_per_second as u128);
    nanos.min((LIFECYCLE_TIMESTAMP_NOT_RECORDED - 1) as u128) as usize
}

#[cfg(windows)]
#[link(name = "Kernel32")]
extern "system" {
    fn QueryPerformanceCounter(performance_count: *mut i64) -> i32;
    fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
}

fn max_inline_capture_sum(words: [usize; MAX_INLINE_CAPTURE_WORDS]) -> usize {
    black_box(words)
        .iter()
        .copied()
        .map(black_box)
        .sum::<usize>()
}

fn oversized_capture_read_one(words: [usize; OVERSIZED_CAPTURE_WORDS]) -> usize {
    let words = black_box(words);
    black_box(words[0] + words.len() - 1)
}

fn oversized_capture_sum(words: [usize; OVERSIZED_CAPTURE_WORDS]) -> usize {
    black_box(words)
        .iter()
        .copied()
        .map(black_box)
        .sum::<usize>()
}

fn direct_oversized_capture_read_one() -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    verify_oversized_captured_ready_value(oversized_capture_read_one(words))
}

fn direct_oversized_captured_sum() -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    verify_oversized_captured_ready_value(oversized_capture_sum(words))
}

fn direct_boxed_oversized_capture_allocate_drop() -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let task = Box::new(move || oversized_capture_sum(words));
    drop(black_box(task));

    verify_ready_value(READY_VALUE)
}

fn direct_boxed_oversized_capture_execute() -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let task = Box::new(move || oversized_capture_sum(words));
    verify_oversized_captured_ready_value(black_box(task)())
}

fn boxed_ready_value() -> Box<dyn FnOnce() -> usize + Send> {
    Box::new(|| black_box(READY_VALUE))
}
