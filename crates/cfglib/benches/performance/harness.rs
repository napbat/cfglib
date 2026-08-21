#[cfg(not(cfglib_bench_alloc))]
use super::Instant;
#[cfg(cfglib_bench_alloc)]
use super::{AtomicBool, AtomicU64, GlobalAlloc, Layout, Ordering};
use super::{Duration, System, black_box};

#[cfg(cfglib_bench_alloc)]
struct CountingAllocator;

#[cfg(cfglib_bench_alloc)]
static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
#[cfg(cfglib_bench_alloc)]
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

#[cfg(cfglib_bench_alloc)]
fn record_allocation(size: usize) {
    let size = size as u64;
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
        PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
    }
}

#[cfg(cfglib_bench_alloc)]
fn record_deallocation(size: usize) {
    LIVE_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
}

#[cfg(cfglib_bench_alloc)]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        record_deallocation(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(ptr, old, new_size) };
        if !pointer.is_null() {
            let old_size = old.size() as u64;
            let new_size = new_size as u64;
            let live = if new_size >= old_size {
                LIVE_BYTES.fetch_add(new_size - old_size, Ordering::Relaxed) + new_size - old_size
            } else {
                LIVE_BYTES.fetch_sub(old_size - new_size, Ordering::Relaxed) - old_size + new_size
            };
            if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
                PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
            }
        }
        pointer
    }
}

#[cfg(cfglib_bench_alloc)]
#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[cfg(not(cfglib_bench_alloc))]
#[global_allocator]
static GLOBAL_ALLOCATOR: System = System;

#[cfg(cfglib_bench_alloc)]
#[derive(Clone, Copy)]
struct AllocationSample {
    allocations: u64,
    allocated_bytes: u64,
    peak_live_bytes: u64,
}

#[cfg(cfglib_bench_alloc)]
fn allocation_sample<T>(mut operation: impl FnMut() -> T) -> AllocationSample {
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);
    let live_before = LIVE_BYTES.load(Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(live_before, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    drop(black_box(operation()));
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        LIVE_BYTES.load(Ordering::Relaxed),
        live_before,
        "benchmark operation changed live allocation bytes"
    );
    AllocationSample {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(live_before),
    }
}

#[cfg(not(cfglib_bench_alloc))]
fn run_iterations<T>(iterations: u64, operation: &mut impl FnMut() -> T) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        drop(black_box(operation()));
    }
    start.elapsed()
}

#[cfg(not(cfglib_bench_alloc))]
pub(super) fn benchmark<T>(name: &str, target: Duration, mut operation: impl FnMut() -> T) {
    let mut iterations = 1_u64;
    loop {
        let elapsed = run_iterations(iterations, &mut operation);
        if elapsed >= target {
            break;
        }

        let elapsed_ns = elapsed.as_nanos().max(1);
        let target_ns = target.as_nanos();
        let scale = u64::try_from(target_ns.div_ceil(elapsed_ns)).unwrap_or(u64::MAX);
        iterations = iterations
            .saturating_mul(scale.clamp(2, 100))
            .max(iterations + 1);
    }

    let mut samples = [0.0_f64; 7];
    for sample in &mut samples {
        *sample =
            run_iterations(iterations, &mut operation).as_secs_f64() * 1e9 / iterations as f64;
    }
    samples.sort_unstable_by(f64::total_cmp);
    let median_ns = samples[samples.len() / 2];
    let minimum_ns = samples[0];

    println!("{name:<36} {median_ns:>12.1} ns/op  min {minimum_ns:>12.1}");
}

#[cfg(cfglib_bench_alloc)]
pub(super) fn benchmark<T>(name: &str, _target: Duration, mut operation: impl FnMut() -> T) {
    // Exercise one unmeasured operation so lazy initialization does not become
    // part of an otherwise steady-state per-operation allocation sample.
    drop(black_box(operation()));
    let allocation = allocation_sample(operation);

    println!(
        "{name:<36} allocs {allocations:>8}  bytes {allocated_bytes:>12}  peak {peak_live_bytes:>10}",
        allocations = allocation.allocations,
        allocated_bytes = allocation.allocated_bytes,
        peak_live_bytes = allocation.peak_live_bytes,
    );
}

pub(super) fn configuration_error(message: &str) -> ! {
    eprintln!("cfglib benchmark configuration error: {message}");
    std::process::exit(2);
}

pub(super) fn benchmark_target() -> (u64, Duration) {
    let target_ms = match std::env::var("CFGLIB_BENCH_MS") {
        Ok(value) => match value.parse::<u64>() {
            Ok(0) => configuration_error("CFGLIB_BENCH_MS must be greater than zero"),
            Ok(value) => value,
            Err(_) => configuration_error("CFGLIB_BENCH_MS must be a positive integer"),
        },
        Err(std::env::VarError::NotPresent) => 75,
        Err(std::env::VarError::NotUnicode(_)) => {
            configuration_error("CFGLIB_BENCH_MS must be valid UTF-8")
        }
    };
    (target_ms, Duration::from_millis(target_ms))
}

pub(super) fn run_semantic_oracle<T>(operation: &mut impl FnMut() -> T, oracle: impl FnOnce(&T)) {
    let result = operation();
    oracle(&result);
    drop(result);
}
