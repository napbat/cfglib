//! Liveness analysis.
//!
//! A **backward** data flow analysis that computes, for each program
//! point, the set of variables whose values may be read in the future
//! before being overwritten.
//!
//! A variable is **live** at a point if there exists a path from that
//! point to a use of the variable with no intervening definition.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ops::Range;

use super::bits::{bit_slot, ones_of};
use super::cfg_view::CfgView;
use super::fixpoint::{Direction, Facts, Problem};
use super::{InstrInfo, VariableId, departs_before_throws, first_unwind_point, is_unwind_edge};
use crate::block::BlockId;

/// The liveness problem.
///
/// A block-level fact cannot separate what an unwind successor reads from
/// what a normal successor reads. Each instruction that
/// [`may_unwind`](InstrInfo::may_unwind) toward an unwind successor therefore
/// keeps everything live after the block live before it. That is sound but
/// coarser than [`Liveness`], which reads unwind successors only before
/// throwing instructions and normal successors only after the block.
pub struct LivenessProblem;

impl<V, Var> Problem<V> for LivenessProblem
where
    V: CfgView,
    V::Inst: InstrInfo<Variable = Var>,
    Var: VariableId,
{
    type Fact = BTreeSet<Var>;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Self::Fact {
        // Nothing is live after program exit.
        BTreeSet::new()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.union(b).cloned().collect()
    }

    /// Backward transfer: `live_in` = uses ∪ (`live_out` − defs).
    ///
    /// Walk the block's instructions in **reverse** to compute the
    /// set of variables live at the block's entry.
    fn transfer(&self, view: &V, block: BlockId, live_out: &Self::Fact) -> Self::Fact {
        backward_block(view, block, live_out.clone(), live_out)
    }
}

/// The liveness problem with caller-supplied live-out seeds.
///
/// A graph says nothing about what leaves it: the value a function hands
/// back, the storage a caller reads again, the state something reached
/// through a non-returning exit observes. Each is live at an exit and
/// invisible to a solve that only reads instructions, so the caller
/// states it here rather than patching a fake reader into the graph.
///
/// The seed of a block joins its live-out set before the block's backward
/// transfer runs, so the solve is otherwise [`LivenessProblem`], including
/// its coarse treatment of unwinds. Seeding a block with successors is
/// meaningful and additive — the seed joins what flows back from them — but
/// a caller normally seeds the blocks with no successors and answers with an
/// empty vector everywhere else.
///
/// The seeds join the transfer, not the meet, so the facts a raw solve
/// returns carry only what flows back from a block's successors.
/// [`Liveness::compute_with_exits`] is the door that joins them back in
/// and is what a consumer should use; solving this problem directly is
/// for a caller that wants the fixpoint facts themselves.
pub struct SeededLivenessProblem<S> {
    live_out: S,
}

impl<S> SeededLivenessProblem<S> {
    /// The liveness problem seeded by `live_out` at every block.
    pub const fn new(live_out: S) -> Self {
        Self { live_out }
    }
}

impl<V, Var, S> Problem<V> for SeededLivenessProblem<S>
where
    V: CfgView,
    V::Inst: InstrInfo<Variable = Var>,
    Var: VariableId,
    S: Fn(BlockId) -> Vec<Var>,
{
    type Fact = BTreeSet<Var>;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Self::Fact {
        // Nothing flows back into a block without successors; whatever
        // leaves the function there arrives as that block's seed.
        BTreeSet::new()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.union(b).cloned().collect()
    }

    /// Backward transfer over `live_out` joined with the block's seed.
    fn transfer(&self, view: &V, block: BlockId, live_out: &Self::Fact) -> Self::Fact {
        let mut live = live_out.clone();
        live.extend((self.live_out)(block));
        backward_block(view, block, live, live_out)
    }
}

/// One block's backward transfer for the block-level problems, from `live`
/// at its end.
///
/// A throwing instruction keeps every variable of `successors` live before
/// it, because an unwind successor's reads are among them.
fn backward_block<V, Var>(
    cfg: &V,
    block: BlockId,
    mut live: BTreeSet<Var>,
    successors: &BTreeSet<Var>,
) -> BTreeSet<Var>
where
    V: CfgView,
    V::Inst: InstrInfo<Variable = Var>,
    Var: VariableId,
{
    let none = BTreeSet::new();
    let on_unwind = if first_unwind_point(cfg, block).is_some() {
        successors
    } else {
        &none
    };
    for instruction in cfg.instructions(block).iter().rev() {
        step_backward(&mut live, instruction, on_unwind);
    }
    live
}

/// One instruction's backward step, from the variables live after it to the
/// variables live before it: kill its defs, then gen its uses.
///
/// A throwing instruction also keeps `on_unwind` live, because an unwind
/// leaves before its definitions complete.
pub(crate) fn step_backward<I: InstrInfo>(
    live: &mut BTreeSet<I::Variable>,
    instruction: &I,
    on_unwind: &BTreeSet<I::Variable>,
) {
    for variable in instruction.defs() {
        live.remove(variable);
    }
    for variable in instruction.uses() {
        live.insert(variable.clone());
    }
    if instruction.may_unwind() {
        live.extend(on_unwind.iter().cloned());
    }
}

/// The variables of one solve, sorted, so that a variable's position is its
/// bit in every row.
struct Universe<V> {
    variables: Vec<V>,
}

impl<V: Ord + Clone> Universe<V> {
    /// The variables that `cfg` reads or writes, and those of `seeds`.
    fn of<G>(cfg: &G, seeds: &[Vec<V>]) -> Self
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let mut variables: Vec<V> = seeds.iter().flatten().cloned().collect();
        for block in cfg.node_ids() {
            for instruction in cfg.instructions(block) {
                variables.extend(instruction.uses().iter().cloned());
                variables.extend(instruction.defs().iter().cloned());
            }
        }
        variables.sort();
        variables.dedup();
        Self { variables }
    }

    /// The words of one row.
    fn words(&self) -> usize {
        self.variables.len().div_ceil(64)
    }

    /// Sets the bit of `variable` in `row`.
    fn insert(&self, row: &mut [u64], variable: &V) {
        if let Ok(index) = self.variables.binary_search(variable) {
            let (word, bit) = bit_slot(index);
            row[word] |= bit;
        }
    }

    /// Clears the bit of `variable` in `row`.
    fn remove(&self, row: &mut [u64], variable: &V) {
        if let Ok(index) = self.variables.binary_search(variable) {
            let (word, bit) = bit_slot(index);
            row[word] &= !bit;
        }
    }

    /// Returns the variables of `row`.
    fn set(&self, row: &[u64]) -> BTreeSet<V> {
        ones_of(row)
            .map(|index| self.variables[index].clone())
            .collect()
    }
}

/// What each block does to liveness, one bit row per block in a flat buffer.
struct BlockRows {
    /// The words of one row.
    words: usize,
    /// The upward-exposed reads of each block.
    reads: Vec<u64>,
    /// The writes of each block.
    writes: Vec<u64>,
    /// The seed of each block.
    seeded: Vec<u64>,
    /// Per block, whether an instruction may unwind. Empty when no edge of
    /// the graph unwinds.
    unwinding: Vec<bool>,
    /// Per block, the writes before the first possible unwind: the only
    /// writes every unwind departure has completed. Empty when no edge of
    /// the graph unwinds.
    before_unwind: Vec<u64>,
}

impl BlockRows {
    fn of<G, Var>(cfg: &G, universe: &Universe<Var>, seeds: &[Vec<Var>], has_unwind: bool) -> Self
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = Var>,
        Var: VariableId,
    {
        let bound = cfg.node_bound();
        let words = universe.words();
        let mut rows = Self {
            words,
            reads: alloc::vec![0; bound * words],
            writes: alloc::vec![0; bound * words],
            seeded: alloc::vec![0; bound * words],
            unwinding: alloc::vec![false; if has_unwind { bound } else { 0 }],
            before_unwind: alloc::vec![0; if has_unwind { bound * words } else { 0 }],
        };
        for block in cfg.node_ids() {
            let index = block.index();
            let range = row(words, index);
            for variable in &seeds[index] {
                universe.insert(&mut rows.seeded[range.clone()], variable);
            }
            if has_unwind {
                for instruction in cfg.instructions(block) {
                    if instruction.may_unwind() {
                        rows.unwinding[index] = true;
                        break;
                    }
                    for variable in instruction.defs() {
                        universe.insert(&mut rows.before_unwind[range.clone()], variable);
                    }
                }
            }
            for instruction in cfg.instructions(block).iter().rev() {
                for variable in instruction.defs() {
                    universe.remove(&mut rows.reads[range.clone()], variable);
                    universe.insert(&mut rows.writes[range.clone()], variable);
                }
                for variable in instruction.uses() {
                    universe.insert(&mut rows.reads[range.clone()], variable);
                }
            }
        }
        rows
    }
}

/// The words of `block`'s row in a flat buffer of rows `words` long.
fn row(words: usize, block: usize) -> Range<usize> {
    block * words..(block + 1) * words
}

/// Solves liveness and returns the facts of every block, with the seed of a
/// block in its live-out set.
///
/// Normal departures use all completed definitions. Unwind departures use
/// only definitions before the first possible unwind. The block's reads
/// join both paths. Bit rows keep the repeated joins allocation-free.
fn solve<V, Var>(cfg: &V, live_out: impl Fn(BlockId) -> Vec<Var>) -> Facts<BTreeSet<Var>>
where
    V: CfgView,
    V::Inst: InstrInfo<Variable = Var>,
    Var: VariableId,
{
    let bound = cfg.node_bound();
    let mut seeds: Vec<Vec<Var>> = alloc::vec![Vec::new(); bound];
    for block in cfg.node_ids() {
        seeds[block.index()] = live_out(block);
    }
    let universe = Universe::of(cfg, &seeds);
    let has_unwind = cfg.edge_ids().any(|edge| is_unwind_edge(cfg.edge(edge)));
    let rows = BlockRows::of(cfg, &universe, &seeds, has_unwind);
    let words = rows.words;

    let mut live_in = alloc::vec![0u64; bound * words];
    let mut flowing_out = alloc::vec![0u64; bound * words];
    let mut queued = alloc::vec![false; bound];
    let mut worklist: Vec<BlockId> = cfg.node_ids().collect();
    for block in &worklist {
        queued[block.index()] = true;
    }
    let mut next = alloc::vec![0u64; words];
    let mut unwind_out = alloc::vec![0u64; if has_unwind { words } else { 0 }];
    let mut steps = 0;
    while let Some(block) = worklist.pop() {
        steps += 1;
        let index = block.index();
        queued[index] = false;
        let out = &mut flowing_out[row(words, index)];
        out.fill(0);
        unwind_out.fill(0);
        for edge in cfg.outgoing(block) {
            let edge = cfg.edge(edge);
            let destination = if has_unwind && rows.unwinding[index] && is_unwind_edge(edge) {
                &mut unwind_out[..]
            } else {
                &mut out[..]
            };
            for (word, from) in destination
                .iter_mut()
                .zip(&live_in[row(words, edge.target().index())])
            {
                *word |= *from;
            }
        }
        for (position, word) in next.iter_mut().enumerate() {
            let at = index * words + position;
            *word = ((out[position] | rows.seeded[at]) & !rows.writes[at]) | rows.reads[at];
            if has_unwind {
                *word |= unwind_out[position] & !rows.before_unwind[at];
            }
        }
        if next[..] != live_in[row(words, index)] {
            live_in[row(words, index)].copy_from_slice(&next);
            for predecessor in cfg.predecessors(block) {
                if !queued[predecessor.index()] {
                    queued[predecessor.index()] = true;
                    worklist.push(predecessor);
                }
            }
        }
    }

    let mut block_in = alloc::vec![BTreeSet::new(); bound];
    let mut block_out = alloc::vec![BTreeSet::new(); bound];
    for block in cfg.node_ids() {
        let index = block.index();
        let range = row(words, index);
        for (word, seed) in flowing_out[range.clone()]
            .iter_mut()
            .zip(&rows.seeded[range.clone()])
        {
            *word |= *seed;
        }
        block_in[index] = universe.set(&live_in[range.clone()]);
        block_out[index] = universe.set(&flowing_out[range]);
    }
    Facts::from_parts(block_in, block_out, steps)
}

/// Result of a liveness analysis with convenient query methods.
///
/// # Examples
///
/// ```
/// # use cfglib::{Cfg, EdgeKind, InstrInfo};
/// # #[derive(Debug, Clone)]
/// # struct Inst { uses: Vec<u16>, defs: Vec<u16> }
/// # impl InstrInfo for Inst {
/// #     type Variable = u16;
/// #     fn uses(&self) -> &[u16] { &self.uses }
/// #     fn defs(&self) -> &[u16] { &self.defs }
/// # }
/// use cfglib::dataflow::liveness::Liveness;
///
/// let mut cfg = Cfg::<Inst>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
///
/// let r0 = 0;
/// cfg.block_mut(b0).push(Inst { uses: vec![], defs: vec![r0] });
/// cfg.block_mut(b1).push(Inst { uses: vec![r0], defs: vec![] });
///
/// let live = Liveness::compute(&cfg);
/// // r0 is live-out of b0 (used in b1).
/// assert!(live.is_live_out(&r0, b0));
/// ```
pub struct Liveness<V> {
    inner: Facts<BTreeSet<V>>,
}

impl<V: VariableId> Liveness<V> {
    /// Run liveness analysis on the given CFG.
    #[must_use]
    pub fn compute<G>(cfg: &G) -> Self
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        Self::compute_with_exits(cfg, |_| Vec::new())
    }

    /// Run liveness seeded with what leaves each block.
    ///
    /// A graph says nothing about what outlives it, so a value a caller
    /// reads back after the function returns is invisible to
    /// [`compute`](Self::compute). `live_out` states it: its variables
    /// are live at the end of the named block, and everything their
    /// definitions read stays live with them. A caller normally answers
    /// for the blocks with no successors and hands back an empty vector
    /// everywhere else.
    ///
    /// Every query on the result accounts for the seed, including
    /// [`live_out`](Self::live_out) and the per-instruction sets.
    #[must_use]
    pub fn compute_with_exits<G>(cfg: &G, live_out: impl Fn(BlockId) -> Vec<V>) -> Self
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        Self {
            inner: solve(cfg, live_out),
        }
    }

    /// Variables live at the **entry** of a block.
    #[must_use]
    pub fn live_in(&self, block: BlockId) -> &BTreeSet<V> {
        self.inner.fact_in(block)
    }

    /// Variables live after normal completion of a block.
    ///
    /// An unwind that leaves before a throwing instruction does not read this
    /// set; [`live_on_unwind`](Self::live_on_unwind) is what it reads.
    #[must_use]
    pub fn live_out(&self, block: BlockId) -> &BTreeSet<V> {
        self.inner.fact_out(block)
    }

    /// Check if a variable is live at a block's entry.
    #[must_use]
    pub fn is_live_in(&self, variable: &V, block: BlockId) -> bool {
        self.live_in(block).contains(variable)
    }

    /// Check if a variable is live after normal completion of a block.
    #[must_use]
    pub fn is_live_out(&self, variable: &V, block: BlockId) -> bool {
        self.live_out(block).contains(variable)
    }

    /// Variables live immediately **before** each instruction of `block`.
    ///
    /// Element `i` is the live set at the point just before instruction `i`;
    /// for a non-empty block the first element equals
    /// [`live_in`](Self::live_in). Computed on demand by replaying the
    /// block's backward transfer from [`live_out`](Self::live_out). An
    /// instruction that may unwind also keeps
    /// [`live_on_unwind`](Self::live_on_unwind) live before it, since the
    /// unwind leaves before its definitions complete.
    #[must_use]
    pub fn live_before_instructions<G>(
        &self,
        cfg: &G,
        block: BlockId,
    ) -> alloc::vec::Vec<BTreeSet<V>>
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let mut points = self.live_points(cfg, block);
        points.pop();
        points
    }

    /// Variables live immediately **after** each instruction of `block`.
    ///
    /// Element `i` is the live set at the point just after instruction `i`
    /// completes normally; for a non-empty block the last element equals
    /// [`live_out`](Self::live_out). A definition at instruction `i` whose
    /// variable is absent from element `i` is a dead store within this
    /// analysis's precision: an unwind out of instruction `i` never observes
    /// that definition.
    #[must_use]
    pub fn live_after_instructions<G>(
        &self,
        cfg: &G,
        block: BlockId,
    ) -> alloc::vec::Vec<BTreeSet<V>>
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let mut points = self.live_points(cfg, block);
        points.remove(0);
        points
    }

    /// The live set at each program point of `block`: element `i` is the set
    /// just before instruction `i`, and the last element is the set after
    /// the last instruction. The point between two instructions is both
    /// the set after one and the set before the next.
    fn live_points<G>(&self, cfg: &G, block: BlockId) -> alloc::vec::Vec<BTreeSet<V>>
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let instructions = cfg.instructions(block);
        let mut points = alloc::vec![BTreeSet::new(); instructions.len() + 1];
        let mut live = self.live_out(block).clone();
        let unwind = self.live_on_unwind(cfg, block);
        points[instructions.len()].clone_from(&live);
        for (index, instruction) in instructions.iter().enumerate().rev() {
            step_backward(&mut live, instruction, &unwind);
            points[index].clone_from(&live);
        }
        points
    }

    /// Variables an unwind out of `block` can read.
    ///
    /// This is the live-in of every unwind successor whose edge departs
    /// before a throwing instruction. Each instruction of `block` that
    /// [`may_unwind`](InstrInfo::may_unwind) keeps the set live before it.
    /// An unwind out of a block that reports no throwing instruction departs
    /// after the block, so its successors are in
    /// [`live_out`](Self::live_out) instead and this set is empty.
    #[must_use]
    pub fn live_on_unwind<G>(&self, cfg: &G, block: BlockId) -> BTreeSet<V>
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let mut live = BTreeSet::new();
        for edge in cfg.outgoing(block) {
            let edge = cfg.edge(edge);
            if departs_before_throws(cfg, edge) {
                live.extend(self.live_in(edge.target()).iter().cloned());
            }
        }
        live
    }

    /// All variables that are live somewhere in the program.
    #[must_use]
    pub fn all_live_variables<G>(&self, cfg: &G) -> BTreeSet<V>
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let mut all = BTreeSet::new();
        for block_id in cfg.node_ids() {
            all.extend(self.live_in(block_id).iter().cloned());
            all.extend(self.live_out(block_id).iter().cloned());
        }
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::cfg::Cfg;
    use crate::dataflow::fixpoint;
    use crate::edge::EdgeKind;
    use crate::flow::FlowEffect;
    use crate::test_util::{df_def as def, df_ff, df_use as use_, df_with_effect};
    use alloc::vec;

    #[test]
    fn liveness_linear_use_makes_live() {
        // bb0: def r0; use r0
        // r0 should be live-in (because use reads it) and NOT live-out
        // (nothing after the block reads it).
        let cfg = CfgBuilder::build(vec![def("def_r0", 0), use_("use_r0", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        // r0 is used in the block → live-in should contain r0
        // (the def kills it, but the use is after the def so
        //  backward: use generates, def kills → net: not live-in
        //  actually: backward walk: use_r0 gens r0, def_r0 kills r0 → live_in = {})
        // But there's nothing after, so live_out = {}
        assert!(!live.is_live_out(&0, cfg.entry()));
    }

    #[test]
    fn liveness_use_without_def_is_live_in() {
        // bb0: use r0 (no def) → r0 should be live-in
        let cfg = CfgBuilder::build(vec![use_("use_r0", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        assert!(live.is_live_in(&0, cfg.entry()));
    }

    #[test]
    fn liveness_dead_def() {
        // bb0: def r0 (never used) → r0 should NOT be live anywhere
        let cfg = CfgBuilder::build(vec![def("def_r0", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        assert!(!live.is_live_in(&0, cfg.entry()));
        assert!(!live.is_live_out(&0, cfg.entry()));
    }

    #[test]
    fn liveness_across_blocks() {
        // bb0: def r0; if
        // bb1: use r0
        // bb2: (nothing)
        // bb3: endif
        // r0 should be live-out of bb0 because bb1 uses it.
        let cfg = CfgBuilder::build(vec![
            def("def_r0", 0),
            df_with_effect(df_ff("if"), FlowEffect::ConditionalOpen),
            use_("use_r0", 0),
            df_with_effect(df_ff("else"), FlowEffect::ConditionalAlternate),
            df_with_effect(df_ff("endif"), FlowEffect::ConditionalClose),
        ])
        .unwrap();
        let live = Liveness::compute(&cfg);
        assert!(
            live.is_live_out(&0, cfg.entry()),
            "r0 is live-out of entry because the true branch uses it"
        );
    }

    #[test]
    fn liveness_empty_single_block() {
        // Empty CFG — no instructions, nothing should be live.
        use crate::test_util::DfInst;
        let cfg = CfgBuilder::build(alloc::vec::Vec::<DfInst>::new()).unwrap();
        let live = Liveness::compute(&cfg);
        assert!(!live.is_live_in(&0, cfg.entry()));
        assert!(!live.is_live_out(&0, cfg.entry()));
    }

    #[test]
    fn liveness_self_loop() {
        // Self-loop: use r0 in a block that loops to itself.
        // r0 is never defined → live-in = {r0}.
        use crate::cfg::Cfg;
        use crate::edge::EdgeKind;
        use crate::test_util::DfInst;

        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(use_("use_r0", 0));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let live = Liveness::compute(&cfg);
        assert!(
            live.is_live_in(&0, cfg.entry()),
            "r0 used but not defined in self-loop"
        );
    }

    #[test]
    fn instruction_granular_sets_refine_the_block_facts() {
        // bb0: def r0; use r0; def r0
        // The final def is a dead store: r0 is not live after it. The first
        // def is live until the use consumes it.
        let cfg =
            CfgBuilder::build(vec![def("first", 0), use_("read", 0), def("dead", 0)]).unwrap();
        let live = Liveness::compute(&cfg);
        let entry = cfg.entry();

        let before = live.live_before_instructions(&cfg, entry);
        let after = live.live_after_instructions(&cfg, entry);
        assert_eq!(before.len(), 3);
        assert_eq!(after.len(), 3);

        assert_eq!(before[0], live.live_in(entry).clone());
        assert_eq!(after[2], live.live_out(entry).clone());

        assert!(after[0].contains(&0), "the first def feeds the use");
        assert!(before[1].contains(&0), "the use reads a live value");
        assert!(!after[2].contains(&0), "the final def is a dead store");
    }

    /// A seed is live-out of the block it names and flows backward from
    /// there like any other live-out fact.
    #[test]
    fn a_seed_is_live_out_of_the_block_it_names() {
        use crate::cfg::Cfg;
        use crate::edge::EdgeKind;
        use crate::test_util::DfInst;

        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(def("def_r0", 0));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let problem = SeededLivenessProblem::new(|block| {
            if block == exit {
                vec![0_u16]
            } else {
                alloc::vec::Vec::new()
            }
        });
        let seeded = fixpoint::solve_problem(&cfg, &problem).unwrap();
        assert!(
            seeded.fact_in(exit).contains(&0),
            "the seed is live where it was stated"
        );
        assert!(
            seeded.fact_out(cfg.entry()).contains(&0),
            "and reaches the definition that produces it"
        );

        assert!(
            !Liveness::compute(&cfg).is_live_out(&0, cfg.entry()),
            "without the seed nothing observes the definition"
        );
    }

    #[test]
    fn liveness_all_live_variables() {
        let cfg = CfgBuilder::build(vec![use_("use_r0", 0), use_("use_r1", 1), def("def_r2", 2)])
            .unwrap();
        let live = Liveness::compute(&cfg);
        let all = live.all_live_variables(&cfg);
        assert!(all.contains(&0));
        assert!(all.contains(&1));
        // r2 is defined but never used, so not live.
        assert!(!all.contains(&2));
    }

    /// A loop that carries one variable around its back edge, with a
    /// variable above 64 so that the rows hold more than one word, and a
    /// seed at the exit. The solve must reach the fixpoint of the problem.
    #[test]
    fn liveness_matches_the_seeded_problem_across_a_loop() {
        use crate::edge::EdgeKind;
        use crate::test_util::{DfInst, df_copy};

        let mut cfg: Cfg<DfInst> = Cfg::new();
        let head = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(def("def_r0", 0));
        cfg.block_mut(cfg.entry()).push(def("def_r70", 70));
        cfg.block_mut(head).push(use_("use_r0", 0));
        cfg.block_mut(body).push(df_copy("copy_r1_r70", 1, 70));
        cfg.block_mut(body).push(df_copy("copy_r0_r1", 0, 1));
        cfg.block_mut(exit).push(def("def_r2", 2));
        cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
        cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, head, EdgeKind::Fallthrough);
        let seed = |block| {
            if block == exit {
                vec![2_u16, 99]
            } else {
                alloc::vec::Vec::new()
            }
        };

        let problem = SeededLivenessProblem::new(seed);
        let expected = fixpoint::solve_problem(&cfg, &problem).unwrap();
        let live = Liveness::compute_with_exits(&cfg, seed);
        for block in cfg.block_ids() {
            assert_eq!(live.live_in(block), expected.fact_in(block), "{block}");
            let mut out = expected.fact_out(block).clone();
            out.extend(seed(block));
            assert_eq!(live.live_out(block), &out, "{block}");
        }
        assert!(live.is_live_in(&70, head), "the loop carries r70");
        assert!(
            live.is_live_in(&99, exit),
            "a seed nothing defines stays live"
        );
    }

    /// `entry → protected`, which falls through to `normal` and unwinds to
    /// `handler`, with `instructions` in `protected`.
    fn protected_region(
        instructions: impl IntoIterator<Item = crate::test_util::DfInst>,
    ) -> (Cfg<crate::test_util::DfInst>, [BlockId; 3]) {
        let mut cfg = Cfg::new();
        let protected = cfg.new_block();
        let normal = cfg.new_block();
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, normal, EdgeKind::Fallthrough);
        cfg.add_edge(protected, handler, EdgeKind::ExceptionUnwind);
        cfg.block_mut(protected)
            .instructions_mut()
            .extend(instructions);
        (cfg, [protected, normal, handler])
    }

    fn call() -> crate::test_util::DfInst {
        df_with_effect(df_ff("call"), FlowEffect::MayThrow)
    }

    #[test]
    fn each_throw_keeps_live_only_the_writes_it_has_completed() {
        let (mut cfg, [protected, _, handler]) = protected_region([
            def("first", 0),
            call(),
            def("second", 0),
            call(),
            def("third", 0),
        ]);
        cfg.block_mut(handler).push(use_("catch", 0));

        let live = Liveness::compute(&cfg);
        let after = live.live_after_instructions(&cfg, protected);
        assert!(after[0].contains(&0), "the first call unwinds with it");
        assert!(after[2].contains(&0), "the second call unwinds with it");
        assert!(!after[4].contains(&0), "no unwind observes the last write");
        assert!(
            !live.is_live_in(&0, protected),
            "every unwind follows a write"
        );
        assert!(
            !live.is_live_out(&0, protected),
            "the normal path never reads it"
        );
        assert_eq!(live.live_on_unwind(&cfg, protected), BTreeSet::from([0]));
    }

    #[test]
    fn a_normal_read_is_not_live_before_a_throw_that_precedes_its_write() {
        let (mut cfg, [protected, normal, _]) = protected_region([call(), def("result", 0)]);
        cfg.block_mut(normal).push(use_("success", 0));

        let live = Liveness::compute(&cfg);
        assert!(live.is_live_out(&0, protected));
        assert!(
            !live.is_live_in(&0, protected),
            "the handler never reads it"
        );
        assert!(live.live_before_instructions(&cfg, protected)[0].is_empty());
    }
}
