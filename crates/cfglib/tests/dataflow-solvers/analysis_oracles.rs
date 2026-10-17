//! Instruction-level oracles: liveness, reaching definitions, and def-use
//! chains computed from the collecting semantics of a program, by exploring
//! its program points directly instead of solving dataflow equations.
//!
//! A program point is `(block, index)`: just before instruction `index`, or
//! the end of the block when `index` is the instruction count. Control moves
//! from point to point exactly as the documentation of unwinding states: an
//! unwind edge leaves just before each throwing instruction of its source (or
//! after the block when it has none), every other edge leaves after the last
//! instruction.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{Cfg, EdgeKind, InstrInfo};

use crate::graphs::{Rng, Shape};

/// The variables a random program reads and writes.
pub const VARIABLES: u8 = 4;

/// One instruction of the toy: what it reads, what it writes, whether it can
/// unwind.
#[derive(Debug, Clone)]
pub struct Ins {
    pub uses: Vec<u8>,
    pub defs: Vec<u8>,
    pub throws: bool,
}

impl Ins {
    pub fn new(uses: &[u8], defs: &[u8]) -> Self {
        Self {
            uses: uses.to_vec(),
            defs: defs.to_vec(),
            throws: false,
        }
    }

    pub fn throwing(uses: &[u8], defs: &[u8]) -> Self {
        Self {
            throws: true,
            ..Self::new(uses, defs)
        }
    }
}

impl InstrInfo for Ins {
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

/// A shape together with the instructions of each block.
pub struct Program {
    pub shape: Shape,
    pub code: Vec<Vec<Ins>>,
}

impl Program {
    pub fn random(rng: &mut Rng, unwinds: bool) -> Self {
        let shape = Shape::random(rng, false, unwinds);
        let subset =
            |rng: &mut Rng| -> Vec<u8> { (0..VARIABLES).filter(|_| rng.below(3) == 0).collect() };
        let code = (0..shape.blocks)
            .map(|_| {
                (0..rng.below(4))
                    .map(|_| Ins {
                        uses: subset(rng),
                        defs: subset(rng),
                        throws: unwinds && rng.below(4) == 0,
                    })
                    .collect()
            })
            .collect();
        Self { shape, code }
    }

    pub fn cfg(&self) -> Cfg<Ins> {
        self.shape.build(self.code.clone())
    }

    fn length(&self, block: usize) -> usize {
        self.code[block].len()
    }

    fn has_throwing_instruction(&self, block: usize) -> bool {
        self.code[block].iter().any(|ins| ins.throws)
    }

    /// The targets an unwind leaves toward just before a throwing
    /// instruction of `block`, and the targets control reaches after the
    /// block completes, as `(before_throw, after_block)`.
    fn departures(&self, block: usize) -> (Vec<usize>, Vec<usize>) {
        let throwing = self.has_throwing_instruction(block);
        let (mut before, mut after) = (Vec::new(), Vec::new());
        for (_, (source, target, kind)) in self.shape.live_edges() {
            if source != block {
                continue;
            }
            if kind == EdgeKind::ExceptionUnwind && throwing {
                before.push(target);
            } else {
                after.push(target);
            }
        }
        (before, after)
    }

    /// Whether `variable` is live at point `(block, index)`: some path from
    /// there reads it before anything writes it. `seeds` names variables
    /// that are also read at the end of a block, by whatever follows it.
    pub fn live_at(
        &self,
        block: usize,
        index: usize,
        variable: u8,
        seeds: &dyn Fn(usize) -> Vec<u8>,
    ) -> bool {
        let mut seen = BTreeSet::new();
        let mut pending = vec![(block, index)];
        while let Some((current, at)) = pending.pop() {
            if !seen.insert((current, at)) {
                continue;
            }
            let (before, after) = self.departures(current);
            if at == self.length(current) {
                if seeds(current).contains(&variable) {
                    return true;
                }
                pending.extend(after.into_iter().map(|target| (target, 0)));
                continue;
            }
            let ins = &self.code[current][at];
            if ins.uses.contains(&variable) {
                return true;
            }
            if ins.throws {
                pending.extend(before.into_iter().map(|target| (target, 0)));
            }
            if !ins.defs.contains(&variable) {
                pending.push((current, at + 1));
            }
        }
        false
    }

    /// The variables live at `(block, index)`.
    pub fn live_set(
        &self,
        block: usize,
        index: usize,
        seeds: &dyn Fn(usize) -> Vec<u8>,
    ) -> BTreeSet<u8> {
        (0..VARIABLES)
            .filter(|variable| self.live_at(block, index, *variable, seeds))
            .collect()
    }

    /// The variables an unwind out of `block` can read: live at the entry of
    /// each target an unwind reaches before a throwing instruction.
    pub fn live_on_unwind(&self, block: usize, seeds: &dyn Fn(usize) -> Vec<u8>) -> BTreeSet<u8> {
        let (before, _) = self.departures(block);
        before
            .into_iter()
            .flat_map(|target| self.live_set(target, 0, seeds))
            .collect()
    }

    /// Every `(point, state)` the program can be in, where the state maps
    /// each variable to the sites that may have written it last.
    pub fn explore(&self) -> Exploration {
        let mut seen: BTreeSet<(usize, usize, State)> = BTreeSet::new();
        let mut pending = vec![(0, 0, State::new())];
        while let Some((block, at, state)) = pending.pop() {
            if !seen.insert((block, at, state.clone())) {
                continue;
            }
            let (before, after) = self.departures(block);
            if at == self.length(block) {
                pending.extend(after.into_iter().map(|target| (target, 0, state.clone())));
                continue;
            }
            let ins = &self.code[block][at];
            if ins.throws {
                pending.extend(before.into_iter().map(|target| (target, 0, state.clone())));
            }
            let mut next = state;
            for variable in &ins.defs {
                next.retain(|(written, _)| written != variable);
            }
            for variable in &ins.defs {
                next.insert((*variable, (block, at)));
            }
            pending.push((block, at + 1, next));
        }
        Exploration { seen }
    }
}

/// The definitions that may be the latest write of each variable.
pub type State = BTreeSet<(u8, (usize, usize))>;

/// The collecting semantics of a program.
pub struct Exploration {
    seen: BTreeSet<(usize, usize, State)>,
}

impl Exploration {
    /// Every state the program can be in at `(block, index)`, merged.
    pub fn reaching(&self, block: usize, index: usize) -> State {
        self.seen
            .iter()
            .filter(|(b, i, _)| *b == block && *i == index)
            .flat_map(|(_, _, state)| state.iter().copied())
            .collect()
    }

    /// For each reachable use site `(block, index)` and variable it reads,
    /// the sites that may have supplied the value.
    pub fn suppliers(
        &self,
        program: &Program,
    ) -> BTreeMap<(usize, usize), BTreeSet<(usize, usize)>> {
        let mut map: BTreeMap<(usize, usize), BTreeSet<(usize, usize)>> = BTreeMap::new();
        for (block, index, state) in &self.seen {
            let Some(ins) = program.code[*block].get(*index) else {
                continue;
            };
            for variable in &ins.uses {
                let entry = map.entry((*block, *index)).or_default();
                entry.extend(
                    state
                        .iter()
                        .filter(|(written, _)| written == variable)
                        .map(|(_, site)| *site),
                );
            }
        }
        map
    }
}
