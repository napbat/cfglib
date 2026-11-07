//! A first-candidate scope query allocates nothing, however deep it climbs.
//!
//! This binary installs a counting allocator that counts only on a thread that
//! asked for it, so the test harness and other tests cannot add to the count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::cmp::Ordering;

use cfglib::{
    EdgeRef, ScopeCursor, ScopeCursorQuery, ScopeDatumId, ScopeEdgeId, ScopeFirstResolution,
    ScopeGraph, ScopeGraphPathLabel, ScopeId, ScopeQuery, ScopeResolutionConfig,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record() {
    if COUNTING.with(Cell::get) {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(pointer, old, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocation requests `operation` makes on this thread.
fn allocations_during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|counting| counting.set(true));
    let result = operation();
    COUNTING.with(|counting| counting.set(false));
    (result, ALLOCATIONS.with(Cell::get))
}

type Names = ScopeGraph<(), (), (), u32, u32>;

/// Lexical lookup of a name through parent edges.
struct Lexical;

impl ScopeCursorQuery<(), (), (), u32, u32> for Lexical {
    type State = ();

    fn initial_state(&self, _: &Names, _: &ScopeQuery<'_, u32>) {}

    fn step(
        &self,
        _: &Names,
        _: &ScopeQuery<'_, u32>,
        _: ScopeCursor,
        (): &(),
        _: EdgeRef<'_, ScopeId, ScopeEdgeId, ()>,
    ) -> Option<()> {
        Some(())
    }

    fn accepts(
        &self,
        graph: &Names,
        input: &ScopeQuery<'_, u32>,
        _: ScopeCursor,
        (): &(),
        datum: ScopeDatumId,
    ) -> bool {
        graph.datum(datum).data() == input.data()
    }

    fn should_extend(
        &self,
        _: &Names,
        _: &ScopeQuery<'_, u32>,
        _: ScopeCursor,
        (): &(),
        accepted_candidate_count: usize,
    ) -> bool {
        accepted_candidate_count == 0
    }

    fn compare_labels(
        &self,
        _: &Names,
        _: &ScopeQuery<'_, u32>,
        left: ScopeGraphPathLabel<'_, ()>,
        right: ScopeGraphPathLabel<'_, ()>,
    ) -> Option<Ordering> {
        let is_edge = |label| matches!(label, ScopeGraphPathLabel::Edge(()));
        Some(is_edge(left).cmp(&is_edge(right)))
    }
}

#[test]
fn first_candidate_query_does_not_allocate() {
    const DEPTH: u32 = 1500;
    let mut graph = Names::new();
    let scopes: Vec<ScopeId> = (0..DEPTH).map(|_| graph.add_scope(())).collect();
    for pair in scopes.windows(2) {
        graph.add_edge(pair[1], pair[0], ());
    }
    // Every scope declares a name, so no scope can be skipped.
    let data: Vec<ScopeDatumId> = (0..DEPTH)
        .zip(&scopes)
        .map(|(name, &scope)| graph.add_datum(scope, (), name))
        .collect();

    let (resolution, allocations) = allocations_during(|| {
        ScopeFirstResolution::try_compute_from(
            &graph,
            scopes[scopes.len() - 1],
            &0,
            &Lexical,
            ScopeResolutionConfig::new(),
        )
    });
    let resolution = resolution.expect("a chain is linear");

    assert_eq!(resolution.candidate(), Some((data[0], scopes[0])));
    assert_eq!(resolution.stats().explored_path_count, scopes.len());
    assert_eq!(allocations, 0);
}
