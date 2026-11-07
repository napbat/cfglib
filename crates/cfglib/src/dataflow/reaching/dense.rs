//! Reaching definitions with a dense bit row as the facts of each block.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::Range;

use super::ReachingDef;
use crate::block::BlockId;
use crate::dataflow::bits::{DenseRow, FlatRows, bit_slot, clear_range, flat_row, set_range};
use crate::dataflow::cfg_view::{CfgView, OperandView};
use crate::dataflow::unwind::is_unwind_edge;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::{TraversalDirection, reachable};
use crate::graph::view::GraphView;
use crate::{DefSite, InstrOperands, VariableId};

/// Result of a reaching definitions analysis with a [`DenseRow`] as the facts
/// of each block.
///
/// The answers are the ones of [`ReachingDefs`](super::ReachingDefs), with the
/// same unwind rule: an unwind predecessor contributes only the definitions
/// completed before its throwing instructions, and a block unreachable from
/// the entry receives and contributes no definition. `ReachingDefs` joins a
/// `BTreeSet` on every edge of every iteration; this solve joins bit rows
/// without allocating and keeps them in two flat buffers, so the facts of
/// every block cost one bit per definition site and no allocation of their
/// own.
///
/// Bit `i` of a row stands for [`definition(i)`](Self::definition). The
/// definitions are every write of every instruction of the graph, sorted as
/// [`ReachingDef`] orders, so the [`ones`](DenseRow::ones) of a row visit the
/// reaching definitions in the order a `BTreeSet` of them iterates.
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
/// use cfglib::{DefSite, DenseReachingDefs};
///
/// let mut cfg = Cfg::<Inst>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.block_mut(b0).push(Inst { uses: vec![], defs: vec![0] });
/// cfg.block_mut(b0).push(Inst { uses: vec![], defs: vec![0] });
/// cfg.block_mut(b1).push(Inst { uses: vec![0], defs: vec![] });
///
/// let rd = DenseReachingDefs::compute(&cfg);
/// assert_eq!(rd.definitions().len(), 2, "one bit per write");
/// let reaching: Vec<DefSite> = rd.reaching_in(b1).ones().map(|bit| rd.definition(bit).site).collect();
/// assert_eq!(reaching, [DefSite { block: b0, inst_idx: 1 }], "the second write kills the first");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseReachingDefs<V> {
    /// The definitions that the bits of a row stand for, sorted.
    definitions: Vec<ReachingDef<V>>,
    /// The definitions reaching the entry of each block slot.
    reaching_in: FlatRows,
    /// The definitions reaching the exit of each block slot.
    reaching_out: FlatRows,
}

impl<V: VariableId> DenseReachingDefs<V> {
    /// Runs reaching definitions on `cfg`.
    ///
    /// `cfg` is a [`CfgView`] whose instructions implement
    /// [`InstrInfo`](crate::InstrInfo), or a
    /// [`WithOperands`](crate::WithOperands) that reads them through a
    /// context, as for [`ReachingDefs::compute`](super::ReachingDefs::compute).
    /// Blocks unreachable from the entry receive no definitions, and
    /// contribute none to the blocks they branch to.
    #[must_use]
    pub fn compute<S>(cfg: &S) -> Self
    where
        S: OperandView<Variable = V>,
    {
        let (operands, cfg) = (cfg.operands(), cfg.view());
        let definitions = definitions_of(cfg, operands);
        let len = definitions.len();
        let has_unwind = cfg.edge_ids().any(|edge| is_unwind_edge(cfg.edge(edge)));
        let rows = BlockRows::of(cfg, operands, &definitions, has_unwind);
        let (reaching_in, reaching_out) = solve(cfg, &rows);
        let bound = cfg.node_bound();
        Self {
            definitions,
            reaching_in: FlatRows::new(len, bound, reaching_in),
            reaching_out: FlatRows::new(len, bound, reaching_out),
        }
    }

    /// The definitions that the bits of a row stand for, in bit order.
    ///
    /// Every row has one bit for each, so the length of this slice is the
    /// universe of every row.
    #[must_use]
    pub fn definitions(&self) -> &[ReachingDef<V>] {
        &self.definitions
    }

    /// The definition that bit `index` of a row stands for.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not below the length of
    /// [`definitions`](Self::definitions).
    #[must_use]
    pub fn definition(&self, index: usize) -> &ReachingDef<V> {
        &self.definitions[index]
    }

    /// The bit that stands for `definition`, or `None` for a write that no
    /// instruction of the graph makes, which never reaches anything.
    #[must_use]
    pub fn index_of(&self, definition: &ReachingDef<V>) -> Option<usize> {
        self.definitions.binary_search(definition).ok()
    }

    /// Definitions reaching the **entry** of `block`.
    ///
    /// An unwind predecessor contributes only the definitions completed
    /// before its throwing instructions.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn reaching_in(&self, block: BlockId) -> DenseRow<'_> {
        self.reaching_in.row(block.index())
    }

    /// Definitions reaching the **exit** of `block` after all of its
    /// instructions complete normally.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn reaching_out(&self, block: BlockId) -> DenseRow<'_> {
        self.reaching_out.row(block.index())
    }

    /// All definitions of `variable` reaching the entry of `block`, ordered
    /// by site.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn defs_of_at_entry(&self, variable: &V, block: BlockId) -> Vec<DefSite> {
        let row = self.reaching_in(block);
        run(&self.definitions, variable)
            .filter(|&index| row.get(index))
            .map(|index| self.definitions[index].site)
            .collect()
    }
}

/// Every definition that an instruction of `cfg` makes, as `operands`
/// reports it, sorted.
fn definitions_of<G, O>(cfg: &G, operands: &O) -> Vec<ReachingDef<O::Variable>>
where
    G: CfgView,
    O: InstrOperands<G::Inst>,
{
    let mut definitions = Vec::new();
    for block in cfg.node_ids() {
        for (inst_idx, instruction) in cfg.instructions(block).iter().enumerate() {
            let site = DefSite { block, inst_idx };
            definitions.extend(
                operands
                    .defs(instruction)
                    .iter()
                    .map(|variable| ReachingDef {
                        variable: variable.clone(),
                        site,
                    }),
            );
        }
    }
    definitions.sort();
    definitions.dedup();
    definitions
}

/// The bits of every definition of `variable`.
///
/// [`ReachingDef`] orders by variable first, so they are one contiguous run
/// of the sorted definitions, ordered by site.
fn run<V: VariableId>(definitions: &[ReachingDef<V>], variable: &V) -> Range<usize> {
    definitions.partition_point(|definition| definition.variable < *variable)
        ..definitions.partition_point(|definition| definition.variable <= *variable)
}

/// What each block does to the definitions that reach it, one bit row per
/// block in a flat buffer.
struct BlockRows {
    /// The words of one row.
    words: usize,
    /// The definitions each block makes that reach its end.
    generated: Vec<u64>,
    /// Every definition of each variable that a block writes.
    killed: Vec<u64>,
    /// Per block, whether an instruction may unwind. Empty when no edge of
    /// the graph unwinds.
    unwinding: Vec<bool>,
    /// Per block, the definitions it makes that reach one of its throwing
    /// instructions. Empty when no edge of the graph unwinds.
    unwind_generated: Vec<u64>,
    /// Per block, every definition of each variable that it writes before
    /// its first throwing instruction. Empty when no edge of the graph
    /// unwinds.
    unwind_killed: Vec<u64>,
}

impl BlockRows {
    /// Replays each block of `cfg` from no definitions.
    ///
    /// The state before a throwing instruction is `G ∪ (in − K)` for the
    /// definitions `G` the block made so far and the runs `K` of the
    /// variables it wrote so far. `K` only grows along the block, so the join
    /// of those states over every throwing instruction is the join of their
    /// `G` together with `in` less the `K` of the first.
    fn of<G, O>(
        cfg: &G,
        operands: &O,
        definitions: &[ReachingDef<O::Variable>],
        has_unwind: bool,
    ) -> Self
    where
        G: CfgView,
        O: InstrOperands<G::Inst>,
    {
        let bound = cfg.node_bound();
        let words = definitions.len().div_ceil(64);
        let unwind_rows = if has_unwind { bound * words } else { 0 };
        let mut rows = Self {
            words,
            generated: alloc::vec![0; bound * words],
            killed: alloc::vec![0; bound * words],
            unwinding: alloc::vec![false; if has_unwind { bound } else { 0 }],
            unwind_generated: alloc::vec![0; unwind_rows],
            unwind_killed: alloc::vec![0; unwind_rows],
        };
        for block in cfg.node_ids() {
            let index = block.index();
            let range = flat_row(words, index);
            for (inst_idx, instruction) in cfg.instructions(block).iter().enumerate() {
                if has_unwind && operands.may_unwind(instruction) {
                    if !rows.unwinding[index] {
                        rows.unwinding[index] = true;
                        rows.unwind_killed[range.clone()]
                            .copy_from_slice(&rows.killed[range.clone()]);
                    }
                    for (word, made) in rows.unwind_generated[range.clone()]
                        .iter_mut()
                        .zip(&rows.generated[range.clone()])
                    {
                        *word |= *made;
                    }
                }
                for variable in operands.defs(instruction) {
                    let run = run(definitions, variable);
                    clear_range(&mut rows.generated[range.clone()], run.clone());
                    set_range(&mut rows.killed[range.clone()], run);
                }
                let site = DefSite { block, inst_idx };
                for variable in operands.defs(instruction) {
                    let run = run(definitions, variable);
                    let at = definitions[run.clone()]
                        .binary_search_by(|definition| definition.site.cmp(&site))
                        .expect("the definitions hold every write of the graph");
                    let (word, bit) = bit_slot(run.start + at);
                    rows.generated[range.start + word] |= bit;
                }
            }
        }
        rows
    }
}

/// Solves reaching definitions over the blocks reachable from the entry and
/// returns the flat `(in, out)` rows of every block slot.
fn solve<G: CfgView>(cfg: &G, rows: &BlockRows) -> (Vec<u64>, Vec<u64>) {
    let bound = cfg.node_bound();
    let words = rows.words;
    let has_unwind = !rows.unwinding.is_empty();
    let reached = reachable(cfg, [cfg.root()], TraversalDirection::Outgoing);
    let mut reaching_in = alloc::vec![0u64; bound * words];
    let mut reaching_out = alloc::vec![0u64; bound * words];
    // A block's out row is `generated` even when nothing reaches its entry,
    // so every reachable block is solved once whether or not its in row
    // changes.
    let mut solved = alloc::vec![false; bound];
    let mut queued = alloc::vec![false; bound];
    let mut worklist: Vec<BlockId> = cfg
        .node_ids()
        .filter(|block| reached[block.index()])
        .collect();
    worklist.reverse();
    for block in &worklist {
        queued[block.index()] = true;
    }
    let mut next = alloc::vec![0u64; words];
    while let Some(block) = worklist.pop() {
        let index = block.index();
        queued[index] = false;
        next.fill(0);
        for edge in cfg.incoming(block) {
            let edge = cfg.edge(edge);
            let source = edge.source().index();
            if !reached[source] {
                continue;
            }
            let from = flat_row(words, source);
            if has_unwind && rows.unwinding[source] && is_unwind_edge(edge) {
                for (position, word) in next.iter_mut().enumerate() {
                    let at = from.start + position;
                    *word |=
                        rows.unwind_generated[at] | (reaching_in[at] & !rows.unwind_killed[at]);
                }
            } else {
                for (word, out) in next.iter_mut().zip(&reaching_out[from]) {
                    *word |= *out;
                }
            }
        }
        let range = flat_row(words, index);
        if solved[index] && next[..] == reaching_in[range.clone()] {
            continue;
        }
        solved[index] = true;
        reaching_in[range.clone()].copy_from_slice(&next);
        for (position, word) in reaching_out[range].iter_mut().enumerate() {
            let at = index * words + position;
            *word = rows.generated[at] | (next[position] & !rows.killed[at]);
        }
        for successor in cfg.successors(block) {
            if !queued[successor.index()] {
                queued[successor.index()] = true;
                worklist.push(successor);
            }
        }
    }
    (reaching_in, reaching_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::dataflow::reaching::ReachingDefs;
    use crate::edge::EdgeKind;
    use crate::test_util::protected_region;
    use crate::test_util::{DfInst, df_def as def, df_ff as ff, df_throwing, df_use as use_};
    use alloc::collections::BTreeSet;

    /// The definitions of a row, through the universe of `rd`.
    fn set(rd: &DenseReachingDefs<u16>, row: DenseRow<'_>) -> BTreeSet<ReachingDef<u16>> {
        row.ones()
            .map(|index| rd.definition(index).clone())
            .collect()
    }

    fn assert_rows_match_the_sets(cfg: &Cfg<DfInst>) -> DenseReachingDefs<u16> {
        let dense = DenseReachingDefs::compute(cfg);
        let sets = ReachingDefs::compute(cfg);
        for block in cfg.block_ids() {
            assert_eq!(
                &set(&dense, dense.reaching_in(block)),
                sets.reaching_in(block),
                "{block}"
            );
            assert_eq!(
                &set(&dense, dense.reaching_out(block)),
                sets.reaching_out(block),
                "{block}"
            );
        }
        dense
    }

    #[test]
    fn a_handler_receives_only_the_definitions_completed_before_each_throw() {
        let (mut cfg, [protected, normal, handler]) = protected_region([
            df_throwing(def("load", 0)),
            def("y", 1),
            df_throwing(ff("call")),
            def("after", 0),
        ]);
        cfg.block_mut(cfg.entry()).push(def("prior", 0));

        let rd = assert_rows_match_the_sets(&cfg);
        let at = |block, inst_idx| DefSite { block, inst_idx };
        assert_eq!(
            rd.defs_of_at_entry(&0, handler),
            [at(cfg.entry(), 0), at(protected, 0)],
            "the faulting load reaches only the later throw; the last write none"
        );
        assert_eq!(rd.defs_of_at_entry(&1, handler), [at(protected, 1)]);
        assert_eq!(rd.defs_of_at_entry(&0, normal), [at(protected, 3)]);
        assert_eq!(rd.defs_of_at_entry(&1, normal), [at(protected, 1)]);
    }

    /// Seventy writes of one variable in a loop, so a run spans two words, and
    /// an unreachable block that writes it too.
    #[test]
    fn a_run_across_words_is_killed_whole() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        let unreachable = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(def("before", 1));
        for _ in 0..70 {
            cfg.block_mut(body).push(def("loop", 0));
        }
        cfg.block_mut(exit).push(use_("read", 0));
        cfg.block_mut(unreachable).push(def("never", 0));
        cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
        cfg.add_edge(body, body, EdgeKind::Back);
        cfg.add_edge(body, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(unreachable, exit, EdgeKind::Fallthrough);

        let rd = assert_rows_match_the_sets(&cfg);
        assert_eq!(rd.definitions().len(), 72);
        let last = DefSite {
            block: body,
            inst_idx: 69,
        };
        assert_eq!(
            rd.defs_of_at_entry(&0, body),
            [last],
            "the back edge carries the last write"
        );
        assert_eq!(rd.defs_of_at_entry(&0, exit), [last]);
        assert_eq!(
            rd.defs_of_at_entry(&1, exit),
            [DefSite {
                block: cfg.entry(),
                inst_idx: 0
            }]
        );
        assert_eq!(
            rd.reaching_out(unreachable).count_ones(),
            0,
            "nothing reaches it"
        );
        let never = ReachingDef {
            variable: 0,
            site: DefSite {
                block: unreachable,
                inst_idx: 0,
            },
        };
        assert!(rd.index_of(&never).is_some(), "every write has a bit");
        assert_eq!(
            rd.index_of(&ReachingDef {
                variable: 2,
                site: last
            }),
            None
        );
    }

    #[test]
    fn a_graph_without_writes_has_empty_rows() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).push(use_("read", 0));
        let rd = assert_rows_match_the_sets(&cfg);
        assert_eq!(rd.definitions(), []);
        assert_eq!(rd.reaching_out(cfg.entry()).len(), 0, "a row has no bit");
        assert_eq!(rd.defs_of_at_entry(&0, cfg.entry()), Vec::new());
    }
}
