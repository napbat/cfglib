//! A toy memory-ISA instruction: ordinary uses and definitions, an ordered
//! list of memory events, and an optional unwind point. Also deterministic
//! random programs over it, as plain data.

use std::collections::BTreeSet;

use cfglib::{
    Cfg, EdgeKind, InstrInfo, MemoryAccess, MemoryAtomicity, MemoryEvent, MemoryEventInfo,
};

/// A consumer-defined fence: ordering strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barrier {
    Acquire,
    Release,
    SequentiallyConsistent,
}

pub type Event<L> = MemoryEvent<L, u8, Barrier>;

/// One instruction: `uses` and `defs` are its ordinary variables, `events`
/// its memory events in semantic order.
#[derive(Debug, Clone)]
pub struct Inst<L> {
    pub uses: Vec<u8>,
    pub defs: Vec<u8>,
    pub events: Vec<Event<L>>,
    pub throws: bool,
}

impl<L> Inst<L> {
    pub fn new(uses: &[u8], defs: &[u8], events: Vec<Event<L>>) -> Self {
        Self {
            uses: uses.to_vec(),
            defs: defs.to_vec(),
            events,
            throws: false,
        }
    }

    /// The same instruction, able to unwind before it runs.
    #[must_use]
    pub fn throwing(mut self) -> Self {
        self.throws = true;
        self
    }

    /// `location = value`.
    pub fn store(location: L, value: u8) -> Self {
        Self::new(
            &[value],
            &[],
            vec![MemoryEvent::Access(MemoryAccess::write(location, [value]))],
        )
    }

    /// `location[base][index] = value`, with `base` and `index` selecting the
    /// sub-location.
    pub fn store_indexed(location: L, base: u8, index: u8, value: u8) -> Self {
        Self::new(
            &[base, index, value],
            &[],
            vec![MemoryEvent::Access(
                MemoryAccess::write(location, [value]).with_address_uses([base, index]),
            )],
        )
    }

    /// `into = location`.
    pub fn load(location: L, into: u8) -> Self {
        Self::new(
            &[],
            &[into],
            vec![MemoryEvent::Access(MemoryAccess::read(location, [into]))],
        )
    }

    /// `old = exchange(location, new)`.
    pub fn exchange(location: L, new: u8, old: u8) -> Self {
        Self::new(
            &[new],
            &[old],
            vec![MemoryEvent::Access(MemoryAccess::read_modify_write(
                location,
                [new],
                [old],
            ))],
        )
    }

    /// A fence and nothing else.
    pub fn fence(barrier: Barrier) -> Self {
        Self::new(&[], &[], vec![MemoryEvent::Fence(barrier)])
    }

    /// A throwing instruction that reports no memory event.
    pub fn call() -> Self {
        Self::new(&[], &[], Vec::new()).throwing()
    }

    /// A pure instruction `def = f(uses)`.
    pub fn compute(def: u8, uses: &[u8]) -> Self {
        Self::new(uses, &[def], Vec::new())
    }
}

impl<L: Clone> Inst<L> {
    /// The same instruction with every access made atomic.
    #[must_use]
    pub fn atomic(mut self) -> Self {
        for event in &mut self.events {
            if let MemoryEvent::Access(access) = event {
                *access = access.clone().with_atomicity(MemoryAtomicity::Atomic);
            }
        }
        self
    }
}

impl<L> InstrInfo for Inst<L> {
    type Variable = u8;

    fn uses(&self) -> &[u8] {
        &self.uses
    }

    fn defs(&self) -> &[u8] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.throws
    }
}

impl<L: Clone + Ord> MemoryEventInfo for Inst<L> {
    type Location = L;
    type Fence = Barrier;

    fn memory_events(&self) -> impl Iterator<Item = Event<L>> {
        self.events.iter().cloned()
    }
}

/// A xorshift generator: the same seed gives the same programs everywhere.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 16;
        usize::try_from(value % u64::try_from(bound).expect("a small bound")).expect("below bound")
    }
}

/// A program as plain data: the shape of the graph and the instructions of
/// each block, so oracles read adjacency from here and not from the store.
#[derive(Clone)]
pub struct Program<L> {
    pub blocks: usize,
    pub edges: Vec<(usize, usize, EdgeKind)>,
    pub hole: Option<usize>,
    pub code: Vec<Vec<Inst<L>>>,
}

/// The variables a random program reads and writes.
pub const VARIABLES: u8 = 4;
/// The locations a random program touches.
pub const LOCATIONS: u8 = 5;

impl Program<u8> {
    /// A random program. Its entry is a branch target whenever some edge
    /// happens to lead back to it, so memory SSA meets the loop-headed entry
    /// as often as any other loop.
    pub fn random(rng: &mut Rng, unwinds: bool) -> Self {
        let blocks = 2 + rng.below(6);
        let mut edges = Vec::new();
        for _ in 0..blocks + rng.below(blocks) {
            let (source, target) = (rng.below(blocks), rng.below(blocks));
            let kind = if unwinds && rng.below(4) == 0 {
                EdgeKind::ExceptionUnwind
            } else {
                EdgeKind::Fallthrough
            };
            edges.push((source, target, kind));
        }
        let hole = (rng.below(4) == 0).then(|| 1 + rng.below(blocks - 1));
        let code = (0..blocks)
            .map(|_| {
                (0..rng.below(4))
                    .map(|_| random_instruction(rng, unwinds))
                    .collect()
            })
            .collect();
        Self {
            blocks,
            edges,
            hole,
            code,
        }
    }
}

fn subset(rng: &mut Rng, of: &[u8]) -> Vec<u8> {
    of.iter().copied().filter(|_| rng.below(2) == 0).collect()
}

fn random_instruction(rng: &mut Rng, unwinds: bool) -> Inst<u8> {
    let variables: Vec<u8> = (0..VARIABLES).collect();
    let uses = subset(rng, &variables);
    let defs = subset(rng, &variables);
    let events = (0..rng.below(3))
        .map(|_| {
            let location = u8::try_from(rng.below(usize::from(LOCATIONS))).expect("small");
            match rng.below(4) {
                0 => MemoryEvent::Access(
                    MemoryAccess::read(location, subset(rng, &defs))
                        .with_address_uses(subset(rng, &uses)),
                ),
                1 => MemoryEvent::Access(
                    MemoryAccess::write(location, subset(rng, &uses))
                        .with_address_uses(subset(rng, &uses)),
                ),
                2 => MemoryEvent::Access(MemoryAccess::read_modify_write(
                    location,
                    subset(rng, &uses),
                    subset(rng, &defs),
                )),
                _ => MemoryEvent::Fence(Barrier::SequentiallyConsistent),
            }
        })
        .collect();
    let mut inst = Inst::new(&uses, &defs, events);
    inst.throws = unwinds && rng.below(4) == 0;
    inst
}

impl Program<u8> {
    /// The locations some live instruction accesses. Only these take part in
    /// the alias classes: a location nobody accesses cannot bridge two that do.
    pub fn accessed_locations(&self) -> BTreeSet<u8> {
        self.code
            .iter()
            .enumerate()
            .filter(|(block, _)| self.is_live(*block))
            .flat_map(|(_, code)| code.iter().flat_map(|inst| inst.events.iter()))
            .filter_map(|event| match event {
                MemoryEvent::Access(access) => Some(*access.location()),
                MemoryEvent::Fence(_) => None,
            })
            .collect()
    }
}

impl<L: Clone> Program<L> {
    /// Build the control-flow graph, then remove the hole.
    pub fn cfg(&self) -> Cfg<Inst<L>> {
        let mut cfg = Cfg::<Inst<L>>::new();
        let mut ids = vec![cfg.entry()];
        ids.extend((1..self.blocks).map(|_| cfg.new_block()));
        for (id, instructions) in ids.iter().zip(&self.code) {
            cfg.block_mut(*id)
                .instructions_mut()
                .extend(instructions.iter().cloned());
        }
        for &(source, target, kind) in &self.edges {
            cfg.add_edge(ids[source], ids[target], kind);
        }
        if let Some(hole) = self.hole {
            assert!(cfg.remove_block(ids[hole]));
        }
        cfg
    }

    /// Whether the entry reaches every live block, so no code is unreachable.
    pub fn every_live_block_is_reachable(&self) -> bool {
        let edges = self.live_edges();
        let mut seen = vec![false; self.blocks];
        let mut stack = vec![0];
        while let Some(block) = stack.pop() {
            if !std::mem::replace(&mut seen[block], true) {
                stack.extend(
                    edges
                        .iter()
                        .filter(|(source, _, _)| *source == block)
                        .map(|(_, target, _)| *target),
                );
            }
        }
        (0..self.blocks).all(|block| !self.is_live(block) || seen[block])
    }

    pub fn is_live(&self, block: usize) -> bool {
        self.hole != Some(block)
    }

    /// The live edges.
    pub fn live_edges(&self) -> Vec<(usize, usize, EdgeKind)> {
        self.edges
            .iter()
            .copied()
            .filter(|(source, target, _)| self.is_live(*source) && self.is_live(*target))
            .collect()
    }
}

/// Blocks in sequence, each falling through to the next.
pub fn chain<L>(blocks: Vec<Vec<Inst<L>>>) -> Cfg<Inst<L>> {
    let mut cfg = Cfg::new();
    let mut previous = cfg.entry();
    for (index, instructions) in blocks.into_iter().enumerate() {
        let block = if index == 0 {
            previous
        } else {
            cfg.new_block()
        };
        cfg.block_mut(block).instructions_mut().extend(instructions);
        if index > 0 {
            cfg.add_edge(previous, block, EdgeKind::Fallthrough);
        }
        previous = block;
    }
    cfg
}

/// A diamond: block 0 branches to blocks 1 and 2, which join in block 3.
pub fn diamond<L>(
    entry: Vec<Inst<L>>,
    left: Vec<Inst<L>>,
    right: Vec<Inst<L>>,
    merge: Vec<Inst<L>>,
) -> Cfg<Inst<L>> {
    let mut cfg = Cfg::new();
    let blocks = [
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    ];
    for (block, instructions) in blocks.iter().zip([entry, left, right, merge]) {
        cfg.block_mut(*block)
            .instructions_mut()
            .extend(instructions);
    }
    cfg.add_edge(blocks[0], blocks[1], EdgeKind::ConditionalTrue);
    cfg.add_edge(blocks[0], blocks[2], EdgeKind::ConditionalFalse);
    cfg.add_edge(blocks[1], blocks[3], EdgeKind::Fallthrough);
    cfg.add_edge(blocks[2], blocks[3], EdgeKind::Fallthrough);
    cfg
}
