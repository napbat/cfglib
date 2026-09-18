//! The two builds this benchmark has, and the one measurement each reports.
//!
//! The default build installs [`System`] directly and reports wall-clock
//! nanoseconds per call. Building with `RUSTFLAGS="--cfg cfglib_bench_alloc"`
//! installs a counting [`GlobalAlloc`] and reports allocation requests and
//! bytes per call instead, so allocator instrumentation never perturbs a
//! timing number that is being compared against another one.

use std::alloc::System;
#[cfg(cfglib_bench_alloc)]
use std::alloc::{GlobalAlloc, Layout};
use std::hint::black_box;
#[cfg(cfglib_bench_alloc)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(cfglib_bench_alloc))]
use std::time::Instant;

#[cfg(cfglib_bench_alloc)]
struct CountingAllocator;

#[cfg(cfglib_bench_alloc)]
static COUNTING: AtomicBool = AtomicBool::new(false);
#[cfg(cfglib_bench_alloc)]
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static REQUESTED_BYTES: AtomicU64 = AtomicU64::new(0);

#[cfg(cfglib_bench_alloc)]
fn record(size: usize) {
    if COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        REQUESTED_BYTES.fetch_add(
            u64::try_from(size).expect("allocator request size fits in u64"),
            Ordering::Relaxed,
        );
    }
}

#[cfg(cfglib_bench_alloc)]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, old, new_size) };
        if !moved.is_null() {
            record(new_size);
        }
        moved
    }
}

#[cfg(cfglib_bench_alloc)]
#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[cfg(not(cfglib_bench_alloc))]
#[global_allocator]
static GLOBAL_ALLOCATOR: System = System;

/// How many calls one sample averages over. The per-call numbers a scratch
/// changes are small integers, so the run has to be long enough that a lazily
/// initialized allocation elsewhere cannot round one of them up.
const CALLS: u64 = 64;

/// Report one case, named for the analysis and the size it ran at.
///
/// Both builds run `operation` `CALLS` times after one unmeasured warm-up
/// call, which is what makes a scratch-taking case a *warm*-scratch number:
/// the buffers grew during the warm-up and every measured call reuses them.
#[cfg(cfglib_bench_alloc)]
pub(crate) fn case<T>(name: &str, mut operation: impl FnMut() -> T) {
    drop(black_box(operation()));
    COUNTING.store(true, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REQUESTED_BYTES.store(0, Ordering::Relaxed);
    for _ in 0..CALLS {
        drop(black_box(operation()));
    }
    COUNTING.store(false, Ordering::Relaxed);

    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes = REQUESTED_BYTES.load(Ordering::Relaxed);
    println!(
        "{name:<34} {:>10} allocs/call {:>12} bytes/call",
        allocations / CALLS,
        bytes / CALLS
    );
}

#[cfg(not(cfglib_bench_alloc))]
pub(crate) fn case<T>(name: &str, mut operation: impl FnMut() -> T) {
    drop(black_box(operation()));
    let mut best = u64::MAX;
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..CALLS {
            drop(black_box(operation()));
        }
        let elapsed =
            u64::try_from(start.elapsed().as_nanos()).expect("elapsed nanoseconds fit in u64");
        best = best.min(elapsed / CALLS);
    }
    println!("{name:<34} {best:>10} ns/call");
}
