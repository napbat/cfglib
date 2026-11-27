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
//! The large table runs three inputs, each in two arrival orders:
//!
//! - `uniform`: uniform random endpoints, and one percent of the nodes carry
//!   a hundred edges each. No numbering of such a graph has more locality
//!   than another, so it is the worst case, and no reordering can help it.
//! - `structured`: procedures laid out one after another, each a flow graph
//!   in the order a builder creates its blocks (fall-through, forward
//!   branches, loop latches), plus calls to procedure entries, half of them
//!   to the one procedure in a hundred that is a hub. Nearby ids are nearby
//!   code, as in a graph built function by function.
//! - `structured_relabeled`: the same edges under a random permutation of
//!   the node ids, so its locality is known and recoverable.
//!
//! `grouped` arrival feeds each source's edges together, as a consumer that
//! builds one function or one file at a time does; `shuffled` feeds one
//! shuffled list, so no phase sees an arrival order a compressed index would
//! find easy. Successor and predecessor scans are both timed, since the
//! dominator, post-dominator, SCC and backward solvers all walk predecessors.
//!
//! Each input's locality is reported as its neighbor-to-neighbor average ID
//! distance (N2N AID, Koohi Esfahani, Kilpatrick and Vandierendonck, IISWC
//! 2021): for a node, the gaps between its consecutive neighbor ids in
//! ascending order, summed and divided by its degree, averaged here over the
//! nodes with two or more neighbors, for successors and predecessors. Lower
//! is more local. It depends on the numbering, not on the arrival order.
//!
//! Each `structured_relabeled` table is followed by an ordered-compaction
//! table for the same arrival: the store compacted in `Insertion`,
//! `BreadthFirst` and `ReversePostorder` order, the searches rooted at the
//! nodes with no predecessor. Each order reports the locality of the
//! numbering it produced, then its compaction, both neighbor walks and
//! `scc_leaves_first` over the result. The `compact` rows, in `Insertion` order,
//! are the baseline: the gain of an order is the drop from them.

#[path = "../tests/common/rng.rs"]
pub mod rng;

use std::alloc::System;
#[cfg(cfglib_bench_alloc)]
use std::alloc::{GlobalAlloc, Layout};
use std::hint::black_box;
#[cfg(cfglib_bench_alloc)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cfglib::{Graph, Id, NodeId, NodeOrder, SccDecomposition, SccNumbering};

use crate::rng::Rng;

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

/// The raw id of the node at `index`.
fn node(index: usize) -> u32 {
    u32::try_from(index).expect("node index fits in u32")
}

const NODES: usize = 1_000_000;
const EDGES: usize = 4_000_000;
/// One percent of the nodes are hubs, and a hub carries a hundred edges.
const HUB_SHARE: usize = 100;
const HUB_DEGREE: usize = 100;
const APPENDS: usize = 100_000;
/// Blocks of one procedure of the structured input: `PROCEDURE_MIN` plus up
/// to `PROCEDURE_SPREAD - 1` more.
const PROCEDURE_MIN: usize = 4;
const PROCEDURE_SPREAD: usize = 60;
/// The farthest a forward branch and a loop latch reach inside a procedure.
const BRANCH_REACH: usize = 7;
const LATCH_REACH: usize = 8;

/// The other scale the same store has to serve: one procedure's flow graph,
/// built and thrown away. Nothing here is large enough to miss cache, so the
/// whole measurement is the fixed per-append cost.
const SMALL_NODES: usize = 500;
const SMALL_EDGES: usize = 1_500;
const SMALL_BUILDS: usize = 1_000;

/// A shuffled edge list with the degree skew of a real symbol graph.
fn edge_list(rng: &mut Rng) -> Vec<(u32, u32)> {
    let hubs = NODES / HUB_SHARE;
    let mut edges = Vec::with_capacity(EDGES);
    for hub in 0..hubs {
        let source = node(hub * HUB_SHARE);
        for _ in 0..HUB_DEGREE {
            let target = node(rng.below(NODES));
            edges.push((source, target));
        }
    }
    while edges.len() < EDGES {
        let source = node(rng.below(NODES));
        let target = node(rng.below(NODES));
        edges.push((source, target));
    }

    // Without the shuffle the hub edges arrive grouped by source, which is
    // the one order a compressed index would find unfairly easy.
    rng.shuffle(&mut edges);
    edges
}

fn append_list(rng: &mut Rng) -> Vec<(u32, u32)> {
    (0..APPENDS)
        .map(|_| {
            let source = node(rng.below(NODES));
            let target = node(rng.below(NODES));
            (source, target)
        })
        .collect()
}

/// Procedures laid out one after another until [`NODES`] blocks exist, each
/// numbered in the order a builder creates its blocks.
///
/// Every block falls through to the next one of its procedure; three blocks
/// in ten also branch forward, one in ten closes a loop back to an earlier
/// block, and one in two calls a procedure entry. Half of the calls go to
/// the one procedure in a hundred that is a hub, the rest to any procedure.
/// The list is grouped by source, in block order.
fn structured_edge_list(rng: &mut Rng) -> Vec<(u32, u32)> {
    let mut procedures = Vec::new();
    let mut laid_out = 0;
    while laid_out < NODES {
        let size = (PROCEDURE_MIN + rng.below(PROCEDURE_SPREAD)).min(NODES - laid_out);
        procedures.push((laid_out, size));
        laid_out += size;
    }
    let hubs: Vec<usize> = procedures
        .iter()
        .step_by(HUB_SHARE)
        .map(|&(entry, _)| entry)
        .collect();

    let mut edges = Vec::new();
    for &(entry, size) in &procedures {
        for offset in 0..size {
            let block = entry + offset;
            let ahead = size - offset - 1;
            if ahead >= 1 {
                edges.push((node(block), node(block + 1)));
            }
            if ahead >= 2 && rng.below(10) < 3 {
                let reach = 2 + rng.below(ahead.min(BRANCH_REACH) - 1);
                edges.push((node(block), node(block + reach)));
            }
            if offset >= 1 && rng.below(10) == 0 {
                let reach = 1 + rng.below(offset.min(LATCH_REACH));
                edges.push((node(block), node(block - reach)));
            }
            if rng.below(2) == 0 {
                let callee = if rng.below(2) == 0 {
                    hubs[rng.below(hubs.len())]
                } else {
                    procedures[rng.below(procedures.len())].0
                };
                edges.push((node(block), node(callee)));
            }
        }
    }
    edges
}

/// `edges` under a random permutation of the node ids.
fn relabeled(edges: &[(u32, u32)], rng: &mut Rng) -> Vec<(u32, u32)> {
    let mut label: Vec<u32> = (0..NODES).map(node).collect();
    rng.shuffle(&mut label);
    edges
        .iter()
        .map(|&(source, target)| (label[source as usize], label[target as usize]))
        .collect()
}

/// The order a configuration feeds its edges to the store.
#[derive(Clone, Copy)]
enum Arrival {
    /// Each source's edges together, in their original order.
    Grouped,
    /// One shuffled list.
    Shuffled,
}

impl Arrival {
    fn name(self) -> &'static str {
        match self {
            Self::Grouped => "grouped",
            Self::Shuffled => "shuffled",
        }
    }

    fn arrange(self, edges: &[(u32, u32)], rng: &mut Rng) -> Vec<(u32, u32)> {
        let mut arranged = edges.to_vec();
        match self {
            // A stable sort keeps each source's edges in their order.
            Self::Grouped => arranged.sort_by_key(|&(source, _)| source),
            Self::Shuffled => rng.shuffle(&mut arranged),
        }
        arranged
    }
}

/// The mean N2N AID of `edges` in thousandths, over the nodes with two or
/// more neighbors, of each node's successors (`outgoing`) or predecessors.
fn n2n_aid_milli(edges: &[(u32, u32)], outgoing: bool) -> u64 {
    let mut pairs: Vec<(u32, u32)> = edges
        .iter()
        .map(|&(source, target)| {
            if outgoing {
                (source, target)
            } else {
                (target, source)
            }
        })
        .collect();
    pairs.sort_unstable();
    let (mut total, mut counted) = (0_u128, 0_u128);
    for run in pairs.chunk_by(|left, right| left.0 == right.0) {
        if run.len() < 2 {
            continue;
        }
        let gaps: u128 = run
            .windows(2)
            .map(|pair| u128::from(pair[1].1 - pair[0].1))
            .sum();
        let degree = u128::try_from(run.len()).expect("a degree fits in u128");
        total += gaps * 1_000 / degree;
        counted += 1;
    }
    u64::try_from(total / counted.max(1)).expect("a mean id distance fits in u64")
}

/// Thousandths to one decimal place, in integers.
fn thousandths(milli: u64) -> String {
    format!("{}.{}", milli / 1_000, (milli % 1_000) / 100)
}

/// A procedure-sized edge list, reused by every repetition so the two small
/// cases build the identical graph a thousand times over.
fn small_edge_list(rng: &mut Rng) -> Vec<(u32, u32)> {
    (0..SMALL_EDGES)
        .map(|_| {
            let source = node(rng.below(SMALL_NODES));
            let target = node(rng.below(SMALL_NODES));
            (source, target)
        })
        .collect()
}

/// A build that states no capacity, which is how a consumer that does not
/// know its block count builds: the growth of every array is part of the
/// measurement.
fn build_small(edges: &[(u32, u32)]) -> Graph<u32, ()> {
    let mut graph = Graph::new();
    for index in 0..SMALL_NODES {
        graph.add_node(node(index));
    }
    for &(source, target) in edges {
        graph.add_edge(Id::from_raw(source), Id::from_raw(target), ());
    }
    graph
}

fn build_store(edges: &[(u32, u32)]) -> Graph<u32, ()> {
    let mut graph = Graph::with_capacity(NODES, edges.len());
    for index in 0..NODES {
        graph.add_node(node(index));
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

/// Read the payload of every node's successors (`outgoing`) or predecessors,
/// the way an analysis reads its neighbors' facts: the numbering of the
/// graph decides how scattered those reads are.
fn walk_neighbors(graph: &Graph<u32, ()>, outgoing: bool) -> usize {
    let mut total = 0_usize;
    for node in graph.node_ids() {
        if outgoing {
            for successor in graph.successors(node) {
                total = total.wrapping_add(*graph.node(successor) as usize);
            }
        } else {
            for predecessor in graph.predecessors(node) {
                total = total.wrapping_add(*graph.node(predecessor) as usize);
            }
        }
    }
    total
}

/// Milliseconds to one decimal place, computed in integers so the report
/// never needs a lossy cast.
fn milliseconds(nanos: u64) -> String {
    format!("{}.{}", nanos / 1_000_000, (nanos % 1_000_000) / 100_000)
}

fn report(run: &[(&str, Sample)]) {
    for (case, sample) in run {
        if *case == "footprint" && !cfg!(cfglib_bench_alloc) {
            continue;
        }
        let millis = milliseconds(sample.nanos);
        if cfg!(cfglib_bench_alloc) {
            println!(
                "{case:<20} {millis:>10} ms {:>10} allocs {:>13} req {:>13} peak {:>13} resident",
                sample.allocations,
                sample.requested_bytes,
                sample.peak_bytes,
                sample.resident_bytes
            );
        } else {
            println!("{case:<20} {millis:>10} ms");
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
    let ((), first_compaction) = measure(|| drop(graph.compact(NodeOrder::Insertion)));
    let (total, scan) = measure(|| walk_neighbors(&graph, true));
    assert!(total > 0, "the scan must observe the whole graph");
    let (reverse_total, reverse_scan) = measure(|| walk_neighbors(&graph, false));
    assert!(reverse_total > 0, "the scan must observe the whole graph");

    let (_, incremental) = measure(|| {
        for &(source, target) in appends {
            graph.add_edge(Id::from_raw(source), Id::from_raw(target), ());
        }
        walk_neighbors(&graph, true)
    });

    let ((), compaction) = measure(|| drop(graph.compact(NodeOrder::Insertion)));
    assert!(graph.is_compact());
    let ((), teardown) = measure(move || drop(graph));
    vec![
        ("build", build),
        ("compact_after_build", first_compaction),
        ("successor_scan", scan),
        ("predecessor_scan", reverse_scan),
        ("append_and_scan", incremental),
        ("compact", compaction),
        ("footprint", footprint(teardown)),
    ]
}

fn run_small(edges: &[(u32, u32)]) -> Vec<(&'static str, Sample)> {
    let (built, build_scan) = measure(|| {
        let mut total = 0_usize;
        for _ in 0..SMALL_BUILDS {
            total = total.wrapping_add(scan_store(&build_small(edges)));
        }
        total
    });
    assert!(built > 0, "the scan must observe the whole graph");

    let (compacted, build_compact_scan) = measure(|| {
        let mut total = 0_usize;
        for _ in 0..SMALL_BUILDS {
            let mut graph = build_small(edges);
            drop(graph.compact(NodeOrder::Insertion));
            total = total.wrapping_add(scan_store(&graph));
        }
        total
    });
    assert_eq!(compacted, built, "compaction preserves every successor");

    vec![
        ("build_scan", build_scan),
        ("build_compact_scan", build_compact_scan),
    ]
}

/// The compactions the ordered table compares, the insertion-order one
/// first: it is the baseline every ordered row is read against.
#[derive(Clone, Copy)]
enum Compaction {
    Plain,
    BreadthFirst,
    ReversePostorder,
}

impl Compaction {
    const ALL: [Self; 3] = [Self::Plain, Self::BreadthFirst, Self::ReversePostorder];

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "compact",
            Self::BreadthFirst => "breadth_first",
            Self::ReversePostorder => "reverse_postorder",
        }
    }

    fn apply(self, graph: &mut Graph<u32, ()>, roots: &[NodeId]) {
        let order = match self {
            Self::Plain => NodeOrder::Insertion,
            Self::BreadthFirst => NodeOrder::BreadthFirst(roots),
            Self::ReversePostorder => NodeOrder::ReversePostorder(roots),
        };
        drop(graph.compact(order));
    }
}

/// The roots the searches start from: the nodes with no predecessor, in slot
/// order. On the structured inputs those are the procedure entries nothing
/// calls, which is what a consumer with no designated entry would hand in.
fn sources(graph: &Graph<u32, ()>) -> Vec<NodeId> {
    graph
        .node_ids()
        .filter(|&node| graph.in_degree(node) == 0)
        .collect()
}

/// A built store compacted `compaction`'s way. The roots are found first,
/// outside any measured region: they are the caller's input, not part of
/// the compaction.
fn build_compacted(edges: &[(u32, u32)], compaction: Compaction) -> (Graph<u32, ()>, Sample) {
    let mut graph = build_store(edges);
    let roots = sources(&graph);
    let ((), compacted) = measure(|| compaction.apply(&mut graph, &roots));
    (graph, compacted)
}

/// Time one compaction and what reads its numbering: both neighbor walks and
/// one whole-graph analysis. The second value is what each read computed,
/// which no numbering may change.
fn run_ordered(
    edges: &[(u32, u32)],
    compaction: Compaction,
) -> (Vec<(&'static str, Sample)>, [usize; 3]) {
    let (graph, compacted) = build_compacted(edges, compaction);
    let (successors, scan) = measure(|| walk_neighbors(&graph, true));
    let (predecessors, reverse_scan) = measure(|| walk_neighbors(&graph, false));
    let (components, scc) =
        measure(|| SccDecomposition::compute(&graph, SccNumbering::LeavesFirst).component_count());
    (
        vec![
            ("compaction", compacted),
            ("successor_scan", scan),
            ("predecessor_scan", reverse_scan),
            ("scc_leaves_first", scc),
        ],
        [successors, predecessors, components],
    )
}

/// Compact `edges` in each order and report what each one costs and buys:
/// the locality of the numbering it produced, and the walks and the SCC
/// decomposition over it.
fn report_ordered(edges: &[(u32, u32)], arrival: Arrival, mode: &str) {
    println!(
        "
graph-store structured_relabeled, {} arrival, compacted in order: {NODES} nodes, {} edges; best of {REPEATS}; mode: {mode}",
        arrival.name(),
        edges.len()
    );
    let mut expected = None;
    for compaction in Compaction::ALL {
        let (graph, _) = build_compacted(edges, compaction);
        let numbered: Vec<(u32, u32)> = graph
            .edges()
            .map(|edge| (edge.source().raw(), edge.target().raw()))
            .collect();
        drop(graph);
        println!(
            "{}: locality {} successors {} predecessors",
            compaction.name(),
            thousandths(n2n_aid_milli(&numbered, true)),
            thousandths(n2n_aid_milli(&numbered, false))
        );
        let runs = (0..REPEATS)
            .map(|_| {
                let (samples, reads) = run_ordered(edges, compaction);
                assert_eq!(
                    *expected.get_or_insert(reads),
                    reads,
                    "a numbering changed what a walk or the SCC decomposition computed"
                );
                samples
            })
            .collect();
        report(&fastest(runs));
    }
}

/// Enough repetitions for the minimum to be meaningful without turning a
/// four-million-edge benchmark into a coffee break.
const REPEATS: usize = 5;

fn main() {
    let mut rng = Rng::xorshift(0x9E37_79B9_7F4A_7C15);
    let uniform = edge_list(&mut rng);
    let appends = append_list(&mut rng);
    let small = small_edge_list(&mut rng);
    // The structured inputs draw from their own generator, so the uniform
    // and procedure-scale inputs stay the ones earlier runs measured.
    let mut structure_rng = Rng::xorshift(0xD1B5_4A32_D192_ED03);
    let structured = structured_edge_list(&mut structure_rng);
    let structured_relabeled = relabeled(&structured, &mut structure_rng);
    let inputs = [
        ("uniform", &uniform),
        ("structured", &structured),
        ("structured_relabeled", &structured_relabeled),
    ];

    let mode = if cfg!(cfglib_bench_alloc) {
        "allocation (timings are perturbed by the counting allocator)"
    } else {
        "cpu"
    };

    println!(
        "locality: N2N AID, mean over the nodes with two or more neighbors; lower is more local"
    );
    let mut locality = Vec::new();
    for (name, edges) in inputs {
        let (successors, predecessors) = (n2n_aid_milli(edges, true), n2n_aid_milli(edges, false));
        println!(
            "{name:<20} {:>12} successors {:>12} predecessors",
            thousandths(successors),
            thousandths(predecessors)
        );
        locality.push((successors, predecessors));
    }
    assert!(
        locality[2].0 > locality[1].0 && locality[2].1 > locality[1].1,
        "relabeling must lose the structured input's locality, or there is nothing to recover"
    );

    for (name, edges) in inputs {
        for arrival in [Arrival::Shuffled, Arrival::Grouped] {
            let arranged = arrival.arrange(edges, &mut rng);
            println!(
                "
graph-store {name}, {} arrival: {NODES} nodes, {} edges, {APPENDS} appends; best of {REPEATS}; mode: {mode}",
                arrival.name(),
                arranged.len()
            );
            let store = fastest(
                (0..REPEATS)
                    .map(|_| run_store(&arranged, &appends))
                    .collect(),
            );
            report(&store);
            if name == "structured_relabeled" {
                report_ordered(&arranged, arrival, mode);
            }
        }
    }

    println!(
        "
graph-store: {SMALL_NODES} nodes, {} edges, {SMALL_BUILDS} builds; best of {REPEATS}; mode: {mode}",
        small.len()
    );
    let procedure = fastest((0..REPEATS).map(|_| run_small(&small)).collect());
    report(&procedure);
}
