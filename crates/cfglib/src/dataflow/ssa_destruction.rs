//! SSA destruction support.
//!
//! Reads the phis of a renamed [`SsaForm`] and reports the copies that would
//! make each phi's operand reach its result along its CFG edge. It rewrites
//! nothing: the consumer materializes the copies in its native instruction
//! representation, sequences copy cycles, and splits critical edges.
//!
//! # Where a copy may be placed
//!
//! A copy belongs to one *edge*: its [`PhiCopy::from`] departure and its
//! [`PhiCopy::to_block`] name it. The departure point is not private to the
//! edge. Emitting the copy at the departure point runs it on every path
//! through that point, so it is correct only when the edge is the sole way
//! out of the point. That fails when the departure block has another
//! successor while the target has other predecessors (a critical edge), and
//! for an [unwind](crate::dataflow::departs_before_throws) departure, which sits
//! before a throwing instruction inside a block that the normal path
//! continues through. The consumer must give such an edge a block of its own,
//! or otherwise place the copy so that it runs only when the edge is taken,
//! before it relies on the copies; the copies of different edges are
//! unrelated parallel assignments even when they share a departure point.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::ssa::{SsaForm, SsaValue};
use crate::dataflow::{ProgramPoint, VariableId};

/// A copy to be materialized on a specific CFG edge.
///
/// The edge is identified by `from` and `to_block`; see the
/// [module documentation](self) for where the copy may be emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiCopy<V> {
    /// Departure point. Copy before this instruction, or at block end.
    pub from: ProgramPoint,
    /// Block containing the lowered phi.
    pub to_block: BlockId,
    /// SSA value defined by the phi.
    pub destination: SsaValue<V>,
    /// SSA value supplied at `from`.
    pub source: SsaValue<V>,
}

/// Compute the copies that realize every phi operand in `ssa`.
///
/// This only reports; it does not eliminate any phi from `ssa` or rewrite
/// anything. There is one copy per phi and departure point: parallel edges
/// from one block to the phi's block share a departure and the value it
/// carries, so they yield one copy rather than a repeated assignment of the
/// same destination.
#[must_use]
pub fn eliminate_phis<V: VariableId>(ssa: &SsaForm<V>) -> Vec<PhiCopy<V>> {
    let mut copies = Vec::new();
    let mut departed = BTreeSet::new();

    for (block, phi) in ssa.phis() {
        departed.clear();
        for (predecessor, source) in phi.operands() {
            if !departed.insert(*predecessor) {
                continue;
            }
            copies.push(PhiCopy {
                from: *predecessor,
                to_block: block,
                destination: phi.result.clone(),
                source: source.clone(),
            });
        }
    }

    copies
}

/// The copies of one edge: a parallel assignment that runs when the edge is
/// taken.
///
/// Produced by [`copies_by_predecessor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiCopyGroup<'a, V> {
    /// Departure point of the edge.
    pub from: ProgramPoint,
    /// Block the edge enters.
    pub to_block: BlockId,
    /// The edge's copies, in the order `copies` listed them. They may need a
    /// temporary when the native instruction representation lowers a cycle.
    pub copies: Vec<&'a PhiCopy<V>>,
}

/// Group phi copies by the edge they belong to, named by the departure point
/// and the target block.
///
/// Two edges that leave the same departure point for different blocks are
/// different groups, because only one of them is taken. Groups are ordered by
/// departure point and then target block.
#[must_use]
pub fn copies_by_predecessor<V>(copies: &[PhiCopy<V>]) -> Vec<PhiCopyGroup<'_, V>> {
    let mut by_edge: BTreeMap<(ProgramPoint, BlockId), Vec<&PhiCopy<V>>> = BTreeMap::new();
    for copy in copies {
        by_edge
            .entry((copy.from, copy.to_block))
            .or_default()
            .push(copy);
    }
    by_edge
        .into_iter()
        .map(|((from, to_block), copies)| PhiCopyGroup {
            from,
            to_block,
            copies,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;

    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_def, df_throwing, df_use, protected_region};
    use alloc::vec;

    #[test]
    fn diamond_phis_lower_to_renamed_copies() {
        let mut cfg = Cfg::<DfInst>::new();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(left).push(df_def("left", 0));
        cfg.block_mut(right).push(df_def("right", 0));
        cfg.block_mut(merge).push(df_use("merged", 0));
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let copies = eliminate_phis(&ssa);
        let merge_copies: Vec<_> = copies
            .iter()
            .filter(|copy| copy.to_block == merge)
            .collect();

        assert_eq!(merge_copies.len(), 2);
        assert!(merge_copies.iter().all(|copy| {
            copy.destination.variable == 0
                && copy.source.variable == 0
                && copy.destination != copy.source
        }));
    }

    /// `entry` feeds `branch`, which leaves for two merge blocks over edges of
    /// kinds `first` and `second`. Each merge block has a second predecessor and
    /// a phi of its own variable, so `branch` is the departure of one copy for
    /// each merge. Returns `(cfg, branch, left_merge, right_merge)`.
    fn two_merges(first: EdgeKind, second: EdgeKind) -> (Cfg<DfInst>, [BlockId; 3]) {
        let mut cfg = Cfg::<DfInst>::new();
        let [branch, left, right, left_side, right_side] =
            core::array::from_fn(|_| cfg.new_block());
        let entry = cfg.entry();
        for target in [branch, left_side, right_side] {
            cfg.add_edge(entry, target, EdgeKind::Fallthrough);
        }
        cfg.add_edge(branch, left, first);
        cfg.add_edge(branch, right, second);
        cfg.add_edge(left_side, left, EdgeKind::Fallthrough);
        cfg.add_edge(right_side, right, EdgeKind::Fallthrough);
        cfg.block_mut(entry).push(df_def("left seed", 0));
        cfg.block_mut(entry).push(df_def("right seed", 1));
        cfg.block_mut(left_side).push(df_def("left side", 0));
        cfg.block_mut(right_side).push(df_def("right side", 1));
        cfg.block_mut(left).push(df_use("left use", 0));
        cfg.block_mut(right).push(df_use("right use", 1));
        (cfg, [branch, left, right])
    }

    #[test]
    fn copies_for_different_successors_of_one_block_form_separate_groups() {
        for second in [EdgeKind::ConditionalFalse, EdgeKind::ExceptionUnwind] {
            let (cfg, [branch, left, right]) = two_merges(EdgeKind::ConditionalTrue, second);
            let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
            let copies = eliminate_phis(&ssa);
            let groups = copies_by_predecessor(&copies);

            // Both copies depart at the end of `branch`: a block with no
            // throwing instruction departs every edge at its instruction count.
            let leaving: Vec<_> = copies
                .iter()
                .filter(|copy| copy.from.block == branch)
                .collect();
            assert_eq!(leaving.len(), 2, "{second:?}");
            assert!(leaving.iter().all(|copy| copy.from.inst_idx == 0));
            let from_branch: Vec<_> = groups
                .iter()
                .filter(|group| group.from.block == branch)
                .collect();
            assert_eq!(from_branch.len(), 2, "one group per edge, {second:?}");
            for group in &groups {
                assert!(
                    group
                        .copies
                        .iter()
                        .all(|copy| { copy.from == group.from && copy.to_block == group.to_block }),
                    "a group is the parallel assignment of one edge: {group:?}"
                );
            }
            let targets: Vec<_> = from_branch.iter().map(|group| group.to_block).collect();
            assert_eq!(targets, [left, right]);
        }
    }

    #[test]
    fn parallel_edges_assign_each_phi_once_per_departure() {
        let mut cfg = Cfg::<DfInst>::new();
        let [source, other, merge] = core::array::from_fn(|_| cfg.new_block());
        let entry = cfg.entry();
        cfg.add_edge(entry, source, EdgeKind::ConditionalTrue);
        cfg.add_edge(entry, other, EdgeKind::ConditionalFalse);
        cfg.add_edge(source, merge, EdgeKind::ConditionalTrue);
        cfg.add_edge(source, merge, EdgeKind::ConditionalFalse);
        cfg.add_edge(other, merge, EdgeKind::Fallthrough);
        cfg.block_mut(source).push(df_def("source", 0));
        cfg.block_mut(other).push(df_def("other", 0));
        cfg.block_mut(merge).push(df_use("merged", 0));

        let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
        assert_eq!(
            ssa.block(merge).phis[0].operands().len(),
            3,
            "the phi keeps one operand per edge"
        );
        let copies = eliminate_phis(&ssa);

        assert_eq!(copies.len(), 2, "two departures, so two copies: {copies:?}");
        for group in copies_by_predecessor(&copies) {
            let mut destinations: Vec<_> =
                group.copies.iter().map(|copy| &copy.destination).collect();
            destinations.sort();
            destinations.dedup();
            assert_eq!(
                destinations.len(),
                group.copies.len(),
                "a parallel assignment writes each destination once"
            );
        }
    }

    #[test]
    fn an_unwind_copy_departs_before_the_throwing_instruction() {
        let (mut cfg, [protected, normal, handler]) =
            protected_region([df_throwing(df_def("load", 0)), df_use("after", 0)]);
        cfg.block_mut(cfg.entry()).push(df_def("prior", 0));
        cfg.block_mut(normal).push(df_use("success", 0));
        cfg.block_mut(handler).push(df_use("catch", 0));

        let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
        let prior = ssa.block(cfg.entry()).instructions[0].defs()[0].clone();
        let copies = eliminate_phis(&ssa);

        let [copy] = copies.as_slice() else {
            panic!("only the handler merges: {copies:?}");
        };
        assert_eq!(copy.to_block, handler);
        assert_eq!(copy.source, prior);
        assert_eq!(
            copy.from,
            ProgramPoint {
                block: protected,
                inst_idx: 0,
            },
            "the faulting load's own definition never reaches the handler"
        );
    }

    #[test]
    fn copies_are_grouped_by_predecessor() {
        let copies = vec![
            PhiCopy {
                from: ProgramPoint {
                    block: BlockId::from_raw(0),
                    inst_idx: 1,
                },
                to_block: BlockId::from_raw(2),
                destination: SsaValue::new(0_u16, 3),
                source: SsaValue::new(0_u16, 1),
            },
            PhiCopy {
                from: ProgramPoint {
                    block: BlockId::from_raw(1),
                    inst_idx: 1,
                },
                to_block: BlockId::from_raw(2),
                destination: SsaValue::new(0_u16, 3),
                source: SsaValue::new(0_u16, 2),
            },
        ];

        let grouped = copies_by_predecessor(&copies);
        assert_eq!(grouped.len(), 2);
        assert!(grouped.iter().all(|group| group.copies.len() == 1));
    }
}
