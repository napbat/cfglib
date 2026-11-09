//! The graph analyses a flow build runs over a whole codebase, as one worker
//! owns them.
//!
//! A worker thread is handed a slice of the corpus and visits one callable at
//! a time. The three analyses it runs per callable each take a scratch, so the
//! worker holds one bundle of them for its whole slice: the scratches grow to
//! the largest callable the worker meets and every callable after that one is
//! analyzed without allocating working storage at all. This test is that
//! worker, and it checks the bundle answers exactly what the allocating entry
//! points do.

use std::thread;

use cfglib::{
    BlockId, Cfg, DominatorScratch, DominatorTree, EdgeKind, FrontierScratch, PostDominatorScratch,
    PostDominatorTree,
};

/// Everything one worker thread keeps between callables.
///
/// This is the shape a consumer wants: one value per worker, created once,
/// passed to each analysis by `&mut`, and never reset by the caller — every
/// `compute_in` and every frontier query clears what it is about to use.
#[derive(Default)]
struct FlowScratch {
    dominators: DominatorScratch,
    post_dominators: PostDominatorScratch,
    frontier: FrontierScratch,
}

/// The results one callable's flow build produces.
struct CallableAnalyses {
    dominators: DominatorTree,
    post_dominators: PostDominatorTree,
    /// The iterated dominance frontier of every block that writes.
    merges: Vec<BlockId>,
}

impl FlowScratch {
    /// Build one callable's flow, reusing every buffer from the last one.
    fn visit(&mut self, cfg: &Cfg<bool>) -> CallableAnalyses {
        let dominators = DominatorTree::compute_in(&mut self.dominators, cfg);
        let post_dominators = PostDominatorTree::compute_in(&mut self.post_dominators, cfg);
        let mut merges = Vec::new();
        dominators.iterated_frontier(cfg, &mut self.frontier, writers(cfg), None, &mut merges);
        CallableAnalyses {
            dominators,
            post_dominators,
            merges,
        }
    }
}

/// The same build with no scratch at all, which is what it has to match.
fn allocating(cfg: &Cfg<bool>) -> CallableAnalyses {
    let dominators = DominatorTree::compute(cfg);
    let post_dominators = PostDominatorTree::compute(cfg);
    let mut merges = Vec::new();
    dominators.iterated_frontier(
        cfg,
        &mut FrontierScratch::new(),
        writers(cfg),
        None,
        &mut merges,
    );
    CallableAnalyses {
        dominators,
        post_dominators,
        merges,
    }
}

/// The blocks whose instruction is a write (`true`).
fn writers(cfg: &Cfg<bool>) -> Vec<BlockId> {
    cfg.block_ids()
        .filter(|&block| cfg.block(block).instructions().contains(&true))
        .collect()
}

fn assert_same(computed: &CallableAnalyses, expected: &CallableAnalyses, label: &str) {
    assert_eq!(computed.dominators, expected.dominators, "{label}");
    assert_eq!(
        computed.post_dominators, expected.post_dominators,
        "{label}"
    );
    assert_eq!(computed.merges, expected.merges, "{label}");
}

/// A callable of `regions` if/else diamonds. Every third region writes in
/// both arms, the others only in the then arm.
fn callable(regions: usize) -> Cfg<bool> {
    let mut cfg = Cfg::new();
    let mut current = cfg.entry();
    for index in 0..regions {
        let then_block = cfg.new_block();
        let else_block = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(current, then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(current, else_block, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
        cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
        cfg.block_mut(then_block).push(true);
        cfg.block_mut(else_block).push(index % 3 == 0);
        cfg.block_mut(merge).push(false);
        current = merge;
    }
    cfg
}

/// A callable whose entry is a loop header: its latch writes and branches
/// back, so the entry is a merge of its own.
fn entry_loop() -> Cfg<bool> {
    let mut cfg = Cfg::new();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    let entry = cfg.entry();
    cfg.add_edge(entry, latch, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(latch, entry, EdgeKind::Back);
    cfg.block_mut(entry).push(false);
    cfg.block_mut(latch).push(true);
    cfg
}

#[test]
fn one_worker_scratch_analyzes_a_whole_slice_of_the_corpus() {
    // Sizes that rise and fall, because a scratch that only ever grows is
    // never asked what it left behind.
    let corpus: Vec<_> = [1, 9, 2, 30, 1, 5, 17, 1]
        .into_iter()
        .map(callable)
        .chain([entry_loop()])
        .collect();

    let mut scratch = FlowScratch::default();
    // Twice over: the second pass runs entirely on buffers already at the
    // slice's high-water mark.
    for pass in 0..2 {
        for cfg in &corpus {
            assert_same(
                &scratch.visit(cfg),
                &allocating(cfg),
                &format!("pass {pass}, {} blocks", cfg.block_count()),
            );
        }
    }
    let looped = scratch.visit(&entry_loop());
    assert_eq!(looped.merges, [entry_loop().entry()]);
}

#[test]
fn a_worker_thread_owns_its_scratch() {
    // The bundle crosses the thread boundary by value, which is the whole
    // point of the scratches being Send.
    let corpus: Vec<_> = [4, 1, 12].into_iter().map(callable).collect();
    let expected: Vec<_> = corpus.iter().map(allocating).collect();

    let handle = thread::spawn(move || {
        let mut scratch = FlowScratch::default();
        for (cfg, expected) in corpus.iter().zip(&expected) {
            assert_same(&scratch.visit(cfg), expected, "worker thread");
        }
        corpus.len()
    });
    assert_eq!(handle.join().expect("the worker thread completes"), 3);
}
