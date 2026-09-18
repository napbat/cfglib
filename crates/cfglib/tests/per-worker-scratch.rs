//! The flow build symtree runs over a whole codebase, as one worker owns it.
//!
//! A worker thread is handed a slice of the corpus and visits one callable at
//! a time. The four analyses it runs per callable each take a scratch, so the
//! worker holds one bundle of them for its whole slice: the scratches grow to
//! the largest callable the worker meets and every callable after that one is
//! analyzed without allocating working storage at all. This test is that
//! worker, and it checks the bundle answers exactly what the allocating entry
//! points do.

use std::thread;

use cfglib::{
    Cfg, DominatorScratch, DominatorTree, EdgeKind, ExactMemoryAlias, InstrInfo, MemoryAccess,
    MemoryEvent, MemoryEventInfo, MemorySSA, MemorySsaScratch, MemoryValueFlow,
    MemoryValueFlowScratch, SsaForm, SsaScratch,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    Local(u8),
    Global,
}

#[derive(Debug, Clone)]
struct Instruction {
    uses: Vec<u16>,
    defs: Vec<u16>,
    events: Vec<MemoryEvent<Slot, u16, ()>>,
}

impl InstrInfo for Instruction {
    type Variable = u16;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }
}

impl MemoryEventInfo for Instruction {
    type Location = Slot;
    type Fence = ();

    fn memory_events(
        &self,
    ) -> impl Iterator<Item = MemoryEvent<Self::Location, Self::Variable, Self::Fence>> {
        self.events.iter().cloned()
    }
}

/// Everything one worker thread keeps between callables.
///
/// This is the shape a consumer wants: one value per worker, created once,
/// passed to each analysis by `&mut`, and never reset by the caller — every
/// `compute_in` clears what it is about to use.
#[derive(Default)]
struct FlowScratch {
    dominators: DominatorScratch,
    ssa: SsaScratch<u16>,
    memory: MemorySsaScratch<Slot, u16, ()>,
    value_flow: MemoryValueFlowScratch<u16>,
}

/// The four results one callable's flow build produces.
struct CallableAnalyses {
    dominators: DominatorTree,
    ssa: SsaForm<u16>,
    memory: MemorySSA<Slot, u16, ()>,
    value_flow: MemoryValueFlow<u16>,
}

impl FlowScratch {
    /// Build one callable's flow, reusing every buffer from the last one.
    fn visit(&mut self, cfg: &Cfg<Instruction>) -> CallableAnalyses {
        let dominators = DominatorTree::compute_in(&mut self.dominators, cfg);
        let ssa = SsaForm::compute_in(&mut self.ssa, cfg, &dominators);
        let memory = MemorySSA::compute_in(&mut self.memory, cfg, &ExactMemoryAlias);
        let value_flow = MemoryValueFlow::compute_in(&mut self.value_flow, &memory, &ssa)
            .expect("the corpus adapter reports only variables its instructions expose");
        CallableAnalyses {
            dominators,
            ssa,
            memory,
            value_flow,
        }
    }
}

/// The same build with no scratch at all, which is what it has to match.
fn allocating(cfg: &Cfg<Instruction>) -> CallableAnalyses {
    let dominators = DominatorTree::compute(cfg);
    let ssa = SsaForm::compute(cfg, &dominators);
    let memory = MemorySSA::compute(cfg, &ExactMemoryAlias);
    let value_flow = MemoryValueFlow::compute(&memory, &ssa)
        .expect("the corpus adapter reports only variables its instructions expose");
    CallableAnalyses {
        dominators,
        ssa,
        memory,
        value_flow,
    }
}

fn assert_same(computed: &CallableAnalyses, expected: &CallableAnalyses, label: &str) {
    assert_eq!(computed.dominators, expected.dominators, "{label}");
    assert_eq!(computed.ssa, expected.ssa, "{label}");
    assert_eq!(computed.memory, expected.memory, "{label}");
    assert_eq!(computed.value_flow, expected.value_flow, "{label}");
}

/// A callable of `regions` if/else diamonds, with register data flow, a
/// stack slot per region, and one global every region touches.
fn callable(regions: usize) -> Cfg<Instruction> {
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

        let base = u16::try_from(index % 5).expect("a small index fits in u16") * 4;
        let slot = Slot::Local(u8::try_from(index % 3).expect("a small index fits in u8"));
        cfg.block_mut(current)
            .push(load(slot.clone(), base, base + 1));
        cfg.block_mut(then_block)
            .push(store(slot.clone(), base + 1));
        cfg.block_mut(else_block)
            .push(store(Slot::Global, base + 1));
        cfg.block_mut(merge).push(load(slot, base, base + 2));
        cfg.block_mut(merge).push(Instruction {
            uses: vec![base + 2],
            defs: vec![base + 3],
            events: Vec::new(),
        });
        current = merge;
    }
    cfg
}

fn load(slot: Slot, address: u16, into: u16) -> Instruction {
    Instruction {
        uses: vec![address],
        defs: vec![into],
        events: vec![MemoryEvent::Access(
            MemoryAccess::read(slot, [into]).with_address_uses([address]),
        )],
    }
}

fn store(slot: Slot, value: u16) -> Instruction {
    Instruction {
        uses: vec![value],
        defs: Vec::new(),
        events: vec![MemoryEvent::Access(MemoryAccess::write(slot, [value]))],
    }
}

#[test]
fn one_worker_scratch_analyzes_a_whole_slice_of_the_corpus() {
    // Sizes that rise and fall, because a scratch that only ever grows is
    // never asked what it left behind.
    let corpus: Vec<_> = [1, 9, 2, 30, 1, 5, 17, 1]
        .into_iter()
        .map(callable)
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
