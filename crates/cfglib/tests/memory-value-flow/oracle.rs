//! Reference computations for memory SSA: alias classes as connected
//! components of the alias relation, and the stores that can reach each
//! access, found by exploring every `(program point, last writes)` state the
//! program can be in. That is the same thing as enumerating every path with
//! loops unrolled until no new state appears, so it needs no path bound.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, EdgeKind, MemoryDefinition, MemoryEvent, MemoryEventSite, MemorySSA, MemorySsaValue,
    ProgramPoint,
};

use crate::machine::Program;

/// The latest write of an alias class: the state it began in, or one event.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Def<L> {
    /// The class's initial state, named by the smallest location in it.
    LiveIn(L),
    /// A write or read/modify/write event.
    Write(MemoryEventSite),
}

/// Map every location to the smallest location of its alias class: the
/// connected component of the symmetric closure of `may_alias`, found by a
/// depth-first walk from each unassigned location in ascending order.
pub fn alias_classes<L: Clone + Ord>(
    locations: &BTreeSet<L>,
    may_alias: &dyn Fn(&L, &L) -> bool,
) -> BTreeMap<L, L> {
    let all: Vec<&L> = locations.iter().collect();
    let mut representative: BTreeMap<L, L> = BTreeMap::new();
    for start in &all {
        if representative.contains_key(*start) {
            continue;
        }
        let mut stack = vec![*start];
        while let Some(current) = stack.pop() {
            if representative
                .insert(current.clone(), (*start).clone())
                .is_some()
            {
                continue;
            }
            for other in &all {
                if !representative.contains_key(*other)
                    && (may_alias(current, other) || may_alias(other, current))
                {
                    stack.push(other);
                }
            }
        }
    }
    representative
}

/// What can reach each access of a program.
pub struct Reaching<L> {
    /// For each read or read/modify/write, the writes that may supply the
    /// state it reads.
    pub reads: BTreeMap<MemoryEventSite, BTreeSet<Def<L>>>,
    /// For each write or read/modify/write, the writes whose state it
    /// replaces.
    pub clobbers: BTreeMap<MemoryEventSite, BTreeSet<Def<L>>>,
}

type State<L> = BTreeMap<L, Def<L>>;

/// Explore every reachable state of `program`. `representative` maps each
/// location to its class.
pub fn explore<L: Clone + Ord>(
    program: &Program<L>,
    representative: &BTreeMap<L, L>,
) -> Reaching<L> {
    let edges = program.live_edges();
    let departures = |block: usize| {
        let throwing = program.code[block].iter().any(|inst| inst.throws);
        let (mut before, mut after) = (Vec::new(), Vec::new());
        for &(source, target, kind) in &edges {
            if source == block {
                if kind == EdgeKind::ExceptionUnwind && throwing {
                    before.push(target);
                } else {
                    after.push(target);
                }
            }
        }
        (before, after)
    };
    let mut reaching = Reaching {
        reads: BTreeMap::new(),
        clobbers: BTreeMap::new(),
    };
    let mut seen: BTreeSet<(usize, usize, State<L>)> = BTreeSet::new();
    let mut pending: Vec<(usize, usize, State<L>)> = vec![(0, 0, State::new())];
    while let Some((block, index, state)) = pending.pop() {
        if !seen.insert((block, index, state.clone())) {
            continue;
        }
        let (before, after) = departures(block);
        if index == program.code[block].len() {
            pending.extend(after.into_iter().map(|target| (target, 0, state.clone())));
            continue;
        }
        let inst = &program.code[block][index];
        if inst.throws {
            pending.extend(before.into_iter().map(|target| (target, 0, state.clone())));
        }
        let point = ProgramPoint {
            block: BlockId::from_index(block),
            inst_idx: index,
        };
        let mut next = state;
        for (event_index, event) in inst.events.iter().enumerate() {
            let MemoryEvent::Access(access) = event else {
                continue;
            };
            let site = MemoryEventSite::new(point, event_index);
            let class = &representative[access.location()];
            let current = next
                .get(class)
                .cloned()
                .unwrap_or_else(|| Def::LiveIn(class.clone()));
            if access.reads() {
                reaching
                    .reads
                    .entry(site)
                    .or_default()
                    .insert(current.clone());
            }
            if access.writes() {
                reaching.clobbers.entry(site).or_default().insert(current);
                next.insert(class.clone(), Def::Write(site));
            }
        }
        pending.push((block, index + 1, next));
    }
    reaching
}

impl<L: Clone + Ord> Reaching<L> {
    /// The reads whose value depends on the write at `write`: a read of a
    /// state it wrote, and, through a read/modify/write that read it, the
    /// reads of what that wrote in turn.
    pub fn transitive_readers(&self, write: MemoryEventSite) -> BTreeSet<MemoryEventSite> {
        let mut sources = BTreeSet::from([write]);
        let mut readers = BTreeSet::new();
        loop {
            let mut changed = false;
            for (site, defs) in &self.reads {
                let fed = defs
                    .iter()
                    .any(|def| matches!(def, Def::Write(source) if sources.contains(source)));
                if fed && readers.insert(*site) {
                    changed = true;
                    if self.clobbers.contains_key(site) {
                        sources.insert(*site);
                    }
                }
            }
            if !changed {
                return readers;
            }
        }
    }
}

/// The concrete definitions a memory-SSA value can hold: its own producer,
/// or, for a phi, those of every operand, each followed once.
pub fn expand<L, V, F>(memory: &MemorySSA<L, V, F>, value: &MemorySsaValue) -> BTreeSet<Def<L>>
where
    L: Clone + Ord,
    V: cfglib::VariableId,
    F: Clone + Eq,
{
    let mut result = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![value.clone()];
    while let Some(current) = pending.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        match memory
            .definition(&current)
            .expect("every memory value has a producer")
        {
            MemoryDefinition::LiveIn { class } => {
                let class = memory.class(*class).expect("a live-in names a class");
                result.insert(Def::LiveIn(class.locations()[0].clone()));
            }
            MemoryDefinition::Event { site } => {
                result.insert(Def::Write(*site));
            }
            MemoryDefinition::Phi { .. } => {
                let phi = memory
                    .phis()
                    .iter()
                    .find(|phi| *phi.result() == current)
                    .expect("a phi value has its phi");
                pending.extend(phi.operands().iter().map(|(_, operand)| operand.clone()));
            }
        }
    }
    result
}
