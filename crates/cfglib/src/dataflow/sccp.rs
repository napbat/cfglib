//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP operates on the generic renamed values in [`SsaForm`] while asking
//! the source instruction adapter to fold native instructions.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::constant_propagation::{ConstValue, ConstantFolder, fold_definitions};
use crate::dataflow::ssa::{SsaForm, SsaInstruction, SsaValue};
use crate::edge::{EdgeId, EdgeKind, KindedEdge};
use crate::{InstrInfo, ProgramPoint, VariableId, departs_before_throws};

/// Result of SCCP analysis.
#[derive(Debug, Clone)]
pub struct SccpAnalysis<V, C> {
    /// Lattice value computed for each renamed SSA value.
    pub values: BTreeMap<SsaValue<V>, ConstValue<C>>,
    /// CFG edges proven executable, by edge identity.
    ///
    /// Edges that join the same pair of blocks are judged apart. A branch arm
    /// the predicate withheld stays out of the set while a parallel edge, or
    /// an unwind into the same block, is in it. A phi operand arrives along
    /// the executable edge it departs on;
    /// [`operand_is_executable`](Self::operand_is_executable) asks exactly
    /// that.
    pub executable_edges: BTreeSet<EdgeId>,
    /// CFG blocks proven reachable.
    pub reachable_blocks: BTreeSet<BlockId>,
}

/// The current lattice value of an SSA value. An absent version-0 value is a
/// live-in — an input the analysis cannot know — so it is `Bottom`, never the
/// optimistic `Top` (which would let a phi fold a runtime-varying input into
/// one arm's constant). Absent positive versions are merely not-yet-computed.
fn lattice_of<V: VariableId, C: Clone + Eq>(
    values: &BTreeMap<SsaValue<V>, ConstValue<C>>,
    value: &SsaValue<V>,
) -> ConstValue<C> {
    values.get(value).cloned().unwrap_or(if value.version == 0 {
        ConstValue::Bottom
    } else {
        ConstValue::Top
    })
}

/// Whether `lower` is at or below `upper` in the lattice the adapter's
/// [`ConstantFolder::meet_constants`] defines.
///
/// Two constants compare through the hook itself: `lower` is below `upper`
/// exactly when meeting the two returns `lower` again.
fn is_below<I: ConstantFolder>(lower: &ConstValue<I::Const>, upper: &ConstValue<I::Const>) -> bool {
    match (lower, upper) {
        (ConstValue::Bottom, _) | (_, ConstValue::Top) => true,
        (ConstValue::Const(lower), ConstValue::Const(upper)) => {
            I::meet_constants(lower, upper).as_ref() == Some(lower)
        }
        _ => false,
    }
}

/// Meet `candidate` into the value of `value`, recording in `changed` that
/// the value moved down the lattice.
fn update_value<I: ConstantFolder>(
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    changed: &mut bool,
    value: &SsaValue<I::Variable>,
    candidate: ConstValue<I::Const>,
) {
    let previous = lattice_of(values, value);
    let next = previous.clone().meet_with(candidate, I::meet_constants);
    debug_assert!(
        is_below::<I>(&next, &previous),
        "a constant meet must stay below the value it refines, or the solve can rise and diverge"
    );
    if next != previous {
        values.insert(value.clone(), next);
        *changed = true;
    }
}

/// The constants known at one instruction's inputs: the entries of its uses
/// that already hold `Const`. Both folding hooks read the same map, so the
/// branch predicate sees exactly what the value folder saw.
fn known_constants<V: VariableId, C: Clone + Eq>(
    annotation: &SsaInstruction<V>,
    values: &BTreeMap<SsaValue<V>, ConstValue<C>>,
) -> BTreeMap<V, C> {
    annotation
        .uses()
        .iter()
        .filter_map(|value| {
            values
                .get(value)
                .and_then(ConstValue::as_const)
                .map(|constant| (value.variable.clone(), constant.clone()))
        })
        .collect()
}

fn evaluate_block<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    changed: &mut bool,
) where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    for (instruction, annotation) in cfg
        .instructions(block)
        .iter()
        .zip(&ssa.block(block).instructions)
    {
        let known = known_constants(annotation, values);
        // The renamer gives each source definition one SSA definition, in
        // the same order, so the two sequences zip.
        for ((_, value), definition) in fold_definitions(instruction, &known).zip(annotation.defs())
        {
            update_value::<I>(values, changed, definition, value);
        }
    }
}

/// The branch outcome the last instruction of `block` proves, or `None` when
/// the block does not end in a decided conditional transfer.
fn decided_branch<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
) -> Option<bool>
where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    let (instruction, annotation) = cfg
        .instructions(block)
        .iter()
        .zip(&ssa.block(block).instructions)
        .next_back()?;
    instruction.fold_branch(&known_constants(annotation, values))
}

/// Queue every outgoing edge of `block` the terminator leaves executable.
///
/// A decided predicate proves exactly one of the two conditional arms, so
/// the opposite arm is withheld. An outgoing edge of any other kind stays
/// executable: the predicate says nothing about an exceptional unwind, a
/// back edge, or any other transfer a frontend attached to the same block.
///
/// Edges are only ever added. A predicate that later lowers to `Bottom`
/// re-derives as undecided and then activates the arm it had withheld, so
/// the executable set stays monotone. Re-derivation happens on every drain
/// scan, so an edge already in `executable` is skipped rather than queued
/// again for a pop that would do nothing.
fn activate_successors<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    executable: &BTreeSet<EdgeId>,
    worklist: &mut Vec<EdgeId>,
) where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    let withheld = match decided_branch(cfg, ssa, block, values) {
        Some(true) => Some(EdgeKind::ConditionalFalse),
        Some(false) => Some(EdgeKind::ConditionalTrue),
        None => None,
    };
    for id in cfg.outgoing(block) {
        let edge = cfg.edge(id);
        if Some(edge.data().kind()) != withheld && !executable.contains(&id) {
            worklist.push(id);
        }
    }
}

/// What the phi operands of one edge have in common: the block they leave
/// and whether they leave before that block's end.
///
/// An edge that [departs before throws](departs_before_throws) delivers the
/// state before each throwing instruction, so each of its operands is named
/// by an instruction index inside the block. Every other edge delivers one
/// operand, named by the block's instruction count. An operand and its edge
/// therefore agree on this pair, and two edges between the same blocks agree
/// on it only when they depart the same way.
type Departure = (BlockId, bool);

/// The [`Departure`] shared by the operands that `edge` delivers.
fn edge_departure<G>(cfg: &G, edge: EdgeId) -> Departure
where
    G: CfgView,
    G::Inst: InstrInfo,
{
    let edge = cfg.edge(edge);
    (edge.source(), departs_before_throws(cfg, edge))
}

/// The [`Departure`] of the edge that delivers the operand named by `point`.
fn operand_departure<G: CfgView>(cfg: &G, point: ProgramPoint) -> Departure {
    (
        point.block,
        point.inst_idx < cfg.instructions(point.block).len(),
    )
}

/// Re-meet every phi of `block` over the operands that arrive along its
/// currently executable incoming edges, and over its start value, which
/// needs no edge: the function starts at the entry whether or not any edge
/// into it ever runs. Called when an edge activates AND from the rescan
/// after a value lowers: a phi whose operand lowers after all its edges are
/// already executable must be re-evaluated, or it keeps a stale
/// over-optimistic constant.
fn evaluate_phis<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    executable: &BTreeSet<EdgeId>,
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    changed: &mut bool,
) where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    let phis = &ssa.block(block).phis;
    if phis.is_empty() {
        return;
    }
    let arriving: BTreeSet<Departure> = cfg
        .incoming(block)
        .filter(|edge| executable.contains(edge))
        .map(|edge| edge_departure(cfg, edge))
        .collect();
    for phi in phis {
        let mut candidate = ConstValue::Top;
        if let Some(start) = phi.start_value() {
            candidate = candidate.meet_with(lattice_of(values, &start), I::meet_constants);
        }
        for (departure, operand) in phi.operands() {
            if arriving.contains(&operand_departure(cfg, *departure)) {
                candidate = candidate.meet_with(lattice_of(values, operand), I::meet_constants);
            }
        }
        update_value::<I>(values, changed, &phi.result, candidate);
    }
}

impl<V: VariableId, C: Clone + Eq> SccpAnalysis<V, C> {
    /// Run sparse conditional constant propagation over a renamed SSA form.
    ///
    /// `ssa` must have been computed from `cfg`. The control-flow adapter
    /// exposes branch predicates through [`ConstantFolder::fold_branch`]: a
    /// block whose last instruction decides its condition activates only the
    /// matching [`EdgeKind::ConditionalTrue`] or
    /// [`EdgeKind::ConditionalFalse`] successor edge. An outgoing edge of any
    /// other kind stays executable, because the predicate says nothing about
    /// it. A block with no decided predicate activates every successor.
    ///
    /// Edges are only ever added, so the result stays monotone: a condition
    /// that lowers to `Bottom` on a later pass activates the arm that was
    /// withheld, and `executable_edges` then names both arms.
    ///
    /// Every live-in is unknown here. Use
    /// [`compute_seeded`](Self::compute_seeded) when the caller knows what
    /// one of them holds.
    ///
    /// An entry block that is a branch target has phis that also merge the
    /// variable's live-in, its [start
    /// value](crate::SsaPhi::start_value). That input is executable from the
    /// start whether or not a back edge ever runs, and it is `Bottom` unless
    /// the variable is seeded.
    #[must_use]
    pub fn compute<G>(cfg: &G, ssa: &SsaForm<V>) -> Self
    where
        G: CfgView,
        G::Inst: ConstantFolder<Const = C> + InstrInfo<Variable = V>,
    {
        Self::compute_seeded(cfg, ssa, &BTreeMap::new())
    }

    /// Run sparse conditional constant propagation with known live-ins.
    ///
    /// `live_ins` names the constant that one variable already holds when
    /// the function starts. The solve seeds each entry as the version-zero
    /// value of that variable, which is what [`SsaForm`] gives a read that
    /// no definition reaches. A live-in has no definition, so nothing in the
    /// solve can lower it: the seed stays for the whole analysis.
    ///
    /// A caller supplies a live-in it knows from outside the function. An
    /// argument the calling convention names is one such value. A hidden
    /// register an observation captured is another. A variable that
    /// `live_ins` does not name stays `Bottom`, as it is in
    /// [`compute`](Self::compute).
    ///
    /// `ssa` must have been computed from `cfg`. Every other rule of
    /// [`compute`](Self::compute) applies without change.
    #[must_use]
    pub fn compute_seeded<G>(cfg: &G, ssa: &SsaForm<V>, live_ins: &BTreeMap<V, C>) -> Self
    where
        G: CfgView,
        G::Inst: ConstantFolder<Const = C> + InstrInfo<Variable = V>,
    {
        let mut values = live_ins
            .iter()
            .map(|(variable, constant)| {
                (
                    SsaValue::live_in(variable.clone()),
                    ConstValue::Const(constant.clone()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut executable_edges = BTreeSet::new();
        let mut reachable_blocks = BTreeSet::new();
        let mut cfg_worklist = Vec::new();
        let mut changed = false;

        reachable_blocks.insert(cfg.root());
        evaluate_phis(
            cfg,
            ssa,
            cfg.root(),
            &executable_edges,
            &mut values,
            &mut changed,
        );
        evaluate_block(cfg, ssa, cfg.root(), &mut values, &mut changed);
        activate_successors(
            cfg,
            ssa,
            cfg.root(),
            &values,
            &executable_edges,
            &mut cfg_worklist,
        );

        while !cfg_worklist.is_empty() || changed {
            while let Some(edge) = cfg_worklist.pop() {
                if !executable_edges.insert(edge) {
                    continue;
                }

                let target = cfg.edge(edge).target();
                let newly_reachable = reachable_blocks.insert(target);
                evaluate_phis(
                    cfg,
                    ssa,
                    target,
                    &executable_edges,
                    &mut values,
                    &mut changed,
                );

                if newly_reachable {
                    evaluate_block(cfg, ssa, target, &mut values, &mut changed);
                    activate_successors(
                        cfg,
                        ssa,
                        target,
                        &values,
                        &executable_edges,
                        &mut cfg_worklist,
                    );
                }
            }

            // Rescan every reachable block once per batch of lowered values.
            // The scan finds its dependents by re-evaluating each block, not
            // from a per-value users index, so it runs once per batch rather
            // than once per lowered value. Changes found during a scan form
            // the next batch, which still handles phis whose operands lower
            // after all incoming edges activated. Each scan also re-derives
            // branch edges, so a predicate that lowers here activates the
            // arm its earlier constant had withheld.
            while changed {
                changed = false;
                for &block in &reachable_blocks {
                    evaluate_phis(
                        cfg,
                        ssa,
                        block,
                        &executable_edges,
                        &mut values,
                        &mut changed,
                    );
                    evaluate_block(cfg, ssa, block, &mut values, &mut changed);
                    activate_successors(
                        cfg,
                        ssa,
                        block,
                        &values,
                        &executable_edges,
                        &mut cfg_worklist,
                    );
                }
            }
        }

        SccpAnalysis {
            values,
            executable_edges,
            reachable_blocks,
        }
    }

    /// Whether the phi operand that departs `departure` reaches `block` along
    /// an executable edge.
    ///
    /// `departure` is the [`ProgramPoint`] that
    /// [`SsaPhi::operands`](crate::SsaPhi::operands) names for the operand.
    /// Two edges can join the same blocks and still depart differently: an
    /// unwind carries the state before a throwing instruction, a branch arm
    /// the state at the end of its source. An operand arrives only along an
    /// executable edge of its own kind of departure, so a withheld arm does
    /// not feed a phi that an executable unwind into the same block reaches.
    /// The solve applies this test to every phi operand. A phi's
    /// [start value](crate::SsaPhi::start_value) has no departure point to
    /// ask about: it is always executable.
    #[must_use]
    pub fn operand_is_executable<G>(&self, cfg: &G, departure: ProgramPoint, block: BlockId) -> bool
    where
        G: CfgView,
        G::Inst: InstrInfo,
    {
        let wanted = operand_departure(cfg, departure);
        cfg.incoming(block).any(|edge| {
            self.executable_edges.contains(&edge) && edge_departure(cfg, edge) == wanted
        })
    }
}

#[cfg(test)]
mod tests;
