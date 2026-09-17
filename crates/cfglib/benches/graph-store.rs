//! The incrementally compacted store at the scale it exists for: a
//! whole-codebase symbol graph.
//!
//! The benchmark has the same two builds as `benches/performance`. The
//! default build installs `System` directly and reports wall-clock time;
//! building with `RUSTFLAGS="--cfg cfglib_bench_alloc"` installs a counting
//! allocator and reports allocation requests, so allocator instrumentation
//! never perturbs a timing result that is being compared.
//!
//! ```powershell
//! cargo bench -p cfglib --bench graph-store
//! $env:RUSTFLAGS = "--cfg cfglib_bench_alloc"; cargo bench -p cfglib --bench graph-store
//! ```
//!
//! The store is built from one shuffled edge list, so no phase sees the
//! grouped-by-source arrival order a compressed index would find unfairly
//! easy, and the degree distribution is skewed the way a real symbol graph's
//! is: one percent of the nodes carry a hundred edges each.

use std::alloc::System;
#[cfg(cfglib_bench_alloc)]
use std::alloc::{GlobalAlloc, Layout};
use std::hint::black_box;
#[cfg(cfglib_bench_alloc)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cfglib::{Graph, Id};

#[cfg(cfglib_bench_alloc)]
struct CountingAllocator;

#[cfg(cfglib_bench_alloc)]
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static REQUESTED_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
#[cfg(cfglib_bench_alloc)]
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

#[cfg(cfglib_bench_alloc)]
fn record_allocation(size: usize) {
    let size = u64::try_from(size).expect("allocator request size fits in u64");
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    REQUESTED_BYTES.fetch_add(size, Ordering::Relaxed);
    PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
}

#[cfg(cfglib_bench_alloc)]
fn record_deallocation(size: usize) {
    let size = u64::try_from(size).expect("allocator request size fits in u64");
    LIVE_BYTES.fetch_sub(size, Ordering::Relaxed);
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

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        record_deallocation(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, old, new_size) };
        if !moved.is_null() {
            record_deallocation(old.size());
            record_allocation(new_size);
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

/// One phase's cost. The allocation fields stay zero in the timing build.
#[derive(Clone, Copy, Default)]
struct Sample {
    nanos: u64,
    allocations: u64,
    requested_bytes: u64,
    peak_bytes: u64,
    resident_bytes: i64,
}

#[cfg(not(cfglib_bench_alloc))]
fn measure<T>(operation: impl FnOnce() -> T) -> (T, Sample) {
    let start = Instant::now();
    let value = black_box(operation());
    let nanos = u64::try_from(start.elapsed().as_nanos()).expect("elapsed nanoseconds fit in u64");
    (
        value,
        Sample {
            nanos,
            ..Sample::default()
        },
    )
}

#[cfg(cfglib_bench_alloc)]
fn measure<T>(operation: impl FnOnce() -> T) -> (T, Sample) {
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let requested = REQUESTED_BYTES.load(Ordering::Relaxed);
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(live, Ordering::Relaxed);

    let start = Instant::now();
    let value = black_box(operation());
    let nanos = u64::try_from(start.elapsed().as_nanos()).expect("elapsed nanoseconds fit in u64");

    let after_live = LIVE_BYTES.load(Ordering::Relaxed);
    let resident = i64::try_from(after_live).expect("live bytes fit in i64")
        - i64::try_from(live).expect("live bytes fit in i64");
    (
        value,
        Sample {
            nanos,
            allocations: ALLOCATIONS.load(Ordering::Relaxed) - allocations,
            requested_bytes: REQUESTED_BYTES.load(Ordering::Relaxed) - requested,
            peak_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed).saturating_sub(live),
            resident_bytes: resident,
        },
    )
}

/// A xorshift generator, so both stores see the same graph on every run.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below bound fits in usize")
    }
}

const NODES: usize = 1_000_000;
const EDGES: usize = 4_000_000;
/// One percent of the nodes are hubs, and a hub carries a hundred edges.
const HUB_SHARE: usize = 100;
const HUB_DEGREE: usize = 100;
const APPENDS: usize = 100_000;

/// A shuffled edge list with the degree skew of a real symbol graph.
fn edge_list(rng: &mut Rng) -> Vec<(u32, u32)> {
    let hubs = NODES / HUB_SHARE;
    let mut edges = Vec::with_capacity(EDGES);
    for hub in 0..hubs {
        let source = u32::try_from(hub * HUB_SHARE).expect("node index fits in u32");
        for _ in 0..HUB_DEGREE {
            let target = u32::try_from(rng.below(NODES)).expect("node index fits in u32");
            edges.push((source, target));
        }
    }
    while edges.len() < EDGES {
        let source = u32::try_from(rng.below(NODES)).expect("node index fits in u32");
        let target = u32::try_from(rng.below(NODES)).expect("node index fits in u32");
        edges.push((source, target));
    }

    // Without the shuffle the hub edges arrive grouped by source, which is
    // the one order a compressed index would find unfairly easy.
    for index in (1..edges.len()).rev() {
        edges.swap(index, rng.below(index + 1));
    }
    edges
}

fn append_list(rng: &mut Rng) -> Vec<(u32, u32)> {
    (0..APPENDS)
        .map(|_| {
            let source = u32::try_from(rng.below(NODES)).expect("node index fits in u32");
            let target = u32::try_from(rng.below(NODES)).expect("node index fits in u32");
            (source, target)
        })
        .collect()
}

fn build_store(edges: &[(u32, u32)]) -> Graph<u32, ()> {
    let mut graph = Graph::with_capacity(NODES, edges.len());
    for node in 0..NODES {
        graph.add_node(u32::try_from(node).expect("node index fits in u32"));
    }
    for &(source, target) in edges {
        graph.add_edge(Id::from_raw(source), Id::from_raw(target), ());
    }
    graph
}

fn scan_store(graph: &Graph<u32, ()>) -> usize {
    let mut total = 0_usize;
    for node in graph.node_ids() {
        for successor in graph.successors(node) {
            total = total.wrapping_add(successor.index());
        }
    }
    total
}

/// Milliseconds to one decimal place, computed in integers so the report
/// never needs a lossy cast.
fn milliseconds(nanos: u64) -> String {
    format!("{}.{}", nanos / 1_000_000, (nanos % 1_000_000) / 100_000)
}

fn report(store: &str, run: &[(&str, Sample)]) {
    for (case, sample) in run {
        if *case == "footprint" && !cfg!(cfglib_bench_alloc) {
            continue;
        }
        let millis = milliseconds(sample.nanos);
        if cfg!(cfglib_bench_alloc) {
            println!(
                "{case:<20} {store:<14} {millis:>10} ms {:>10} allocs {:>13} req {:>13} peak {:>13} resident",
                sample.allocations,
                sample.requested_bytes,
                sample.peak_bytes,
                sample.resident_bytes
            );
        } else {
            println!("{case:<20} {store:<14} {millis:>10} ms");
        }
    }
}

/// The bytes a finished store was holding, read from the drop that released
/// them. Timing a teardown is not interesting, so only the size survives.
fn footprint(teardown: Sample) -> Sample {
    Sample {
        resident_bytes: -teardown.resident_bytes,
        ..Sample::default()
    }
}

/// The fastest observation of each case, which is the statistic least
/// polluted by whatever else the machine was doing.
fn fastest(runs: Vec<Vec<(&'static str, Sample)>>) -> Vec<(&'static str, Sample)> {
    let mut best = runs.into_iter().reduce(|mut best, run| {
        for (slot, (_, sample)) in run.into_iter().enumerate() {
            if sample.nanos < best[slot].1.nanos {
                best[slot].1 = sample;
            }
        }
        best
    });
    best.take().expect("at least one run")
}

fn run_store(edges: &[(u32, u32)], appends: &[(u32, u32)]) -> Vec<(&'static str, Sample)> {
    let (mut graph, build) = measure(|| build_store(edges));
    // The renumbering is dropped inside the measured region so the phase
    // reports the store's footprint, not the store plus a report the caller
    // may or may not keep.
    let ((), first_compaction) = measure(|| drop(graph.compact()));
    let (total, scan) = measure(|| scan_store(&graph));
    assert!(total > 0, "the scan must observe the whole graph");

    let (_, incremental) = measure(|| {
        for &(source, target) in appends {
            graph.add_edge(Id::from_raw(source), Id::from_raw(target), ());
        }
        scan_store(&graph)
    });

    let ((), compaction) = measure(|| drop(graph.compact()));
    assert!(graph.is_compact());
    let ((), teardown) = measure(move || drop(graph));
    vec![
        ("build", build),
        ("compact_after_build", first_compaction),
        ("successor_scan", scan),
        ("append_and_scan", incremental),
        ("compact", compaction),
        ("footprint", footprint(teardown)),
    ]
}

/// Enough repetitions for the minimum to be meaningful without turning a
/// four-million-edge benchmark into a coffee break.
const REPEATS: usize = 5;

fn main() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let edges = edge_list(&mut rng);
    let appends = append_list(&mut rng);

    let mode = if cfg!(cfglib_bench_alloc) {
        "allocation (timings are perturbed by the counting allocator)"
    } else {
        "cpu"
    };
    println!(
        "graph-store: {NODES} nodes, {} edges, {APPENDS} appends; best of {REPEATS}; mode: {mode}",
        edges.len()
    );

    let store = fastest((0..REPEATS).map(|_| run_store(&edges, &appends)).collect());
    report("Graph", &store);
}
