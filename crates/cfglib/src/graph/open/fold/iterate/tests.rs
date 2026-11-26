extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::super::tests::{Digest, random_dag};
use super::super::{
    FoldEnter, MarkScope, OpenFold, OpenFoldConfig, OpenFoldError, OpenFoldMemo, Revisit,
    open_fold, open_fold_in,
};
use crate::test_util::rng::Rng;

/// Iterates without a depth bound or a round limit.
const ITERATE: OpenFoldConfig = OpenFoldConfig {
    max_depth: None,
    revisit: Revisit::Iterate { max_rounds: None },
};

/// Iterates under the round limit `max_rounds`.
const fn round_limit(max_rounds: usize) -> OpenFoldConfig {
    OpenFoldConfig {
        max_depth: None,
        revisit: Revisit::Iterate {
            max_rounds: Some(max_rounds),
        },
    }
}

/// How a module re-exports another.
enum Export {
    /// `export * from "..."`
    All,
    /// `export { from as to } from "..."`
    Renamed(&'static str, &'static str),
}

/// The names each module exports: its own, and what its re-exports bring.
struct Exports<'a> {
    own: &'a [&'a [&'static str]],
    reexports: &'a [&'a [(usize, Export)]],
    entered: Vec<usize>,
}

impl<'a> Exports<'a> {
    fn new(own: &'a [&'a [&'static str]], reexports: &'a [&'a [(usize, Export)]]) -> Self {
        Self {
            own,
            reexports,
            entered: Vec::new(),
        }
    }
}

type Names = BTreeSet<&'static str>;

impl OpenFold for Exports<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = Names;
    type Accumulator = (usize, Names);

    fn successors(&mut self, module: &usize, out: &mut Vec<usize>) {
        out.extend(self.reexports[*module].iter().map(|(target, _)| *target));
    }

    fn mark(&mut self, module: &usize) -> Option<usize> {
        Some(*module)
    }

    fn enter(&mut self, module: &usize) -> FoldEnter<Names, (usize, Names)> {
        self.entered.push(*module);
        FoldEnter::Fold {
            accumulator: (*module, self.own[*module].iter().copied().collect()),
            marks: MarkScope::Isolated,
        }
    }

    fn absorb(
        &mut self,
        (module, names): &mut (usize, Names),
        target: &usize,
        value: Option<Names>,
    ) -> ControlFlow<()> {
        let exported = value.unwrap_or_default();
        for (reexported, export) in self.reexports[*module] {
            if reexported != target {
                continue;
            }
            match export {
                Export::All => names.extend(exported.iter().copied()),
                Export::Renamed(from, to) => {
                    if exported.contains(from) {
                        names.insert(to);
                    }
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _module: &usize, (_, names): (usize, Names)) -> Option<Names> {
        Some(names)
    }
}

#[test]
fn a_reexport_cycle_reaches_the_least_fixpoint_from_either_module() {
    // A: `export const x`, `export { x as z } from "./B"`; B: `export * from
    // "./A"`. The least fixpoint is A = B = {x, z}.
    let own: &[&[&str]] = &[&["x"], &[]];
    let reexports: &[&[(usize, Export)]] =
        &[&[(1, Export::Renamed("x", "z"))], &[(0, Export::All)]];
    let both = Names::from(["x", "z"]);

    // One pass refuses A under B, so B brings nothing back to A; rooted at B,
    // A's rename finds no `x` in B either. That is the documented behavior of
    // `Revisit::Refuse`.
    for root in [0, 1] {
        let mut exports = Exports::new(own, reexports);
        assert_eq!(
            open_fold(&mut exports, root, OpenFoldConfig::default()),
            Ok(Some(Names::from(["x"]))),
            "root {root}",
        );
    }

    for root in [0, 1] {
        let mut exports = Exports::new(own, reexports);
        assert_eq!(
            open_fold(&mut exports, root, ITERATE),
            Ok(Some(both.clone())),
            "root {root}",
        );
        // Each module folds once in the first pass and once per round. Rooted
        // at A, two rounds change an answer and a third confirms them;
        // rooted at B, the first round already reaches the fixpoint.
        assert_eq!(exports.entered.len(), [8, 6][root], "root {root}");
    }
}

#[test]
fn iterate_stores_the_final_answers_and_reuses_them() {
    // The re-export cycle of A and B, rooted at A: both answers become final
    // and go to the table, so a later call rooted at B folds nothing.
    let own: &[&[&str]] = &[&["x"], &[]];
    let reexports: &[&[(usize, Export)]] =
        &[&[(1, Export::Renamed("x", "z"))], &[(0, Export::All)]];
    let both = Names::from(["x", "z"]);
    let mut memo = OpenFoldMemo::default();
    let mut exports = Exports::new(own, reexports);
    assert_eq!(
        open_fold_in(&mut memo, &mut exports, 0, ITERATE),
        Ok(Some(both.clone())),
    );
    assert_eq!(memo.answer_count(), 2);
    assert_eq!(memo.get(&1), Some(Some(&both)));
    let folds = exports.entered.len();
    assert_eq!(
        open_fold_in(&mut memo, &mut exports, 1, ITERATE),
        Ok(Some(both)),
    );
    assert_eq!(exports.entered.len(), folds);
}

#[test]
fn fixpoint_and_memoized_folds_agree_on_random_dags() {
    let mut random = Rng::new(0x94_5EED);
    for _ in 0..500 {
        let (children, unmarked) = random_dag(&mut random);
        for root in 0..children.len() {
            let mut memoized = Digest {
                children: &children,
                unmarked: &unmarked,
                entered: Vec::new(),
            };
            let mut fixpoint = Digest {
                children: &children,
                unmarked: &unmarked,
                entered: Vec::new(),
            };
            assert_eq!(
                open_fold(&mut fixpoint, root, ITERATE),
                open_fold(
                    &mut memoized,
                    root,
                    OpenFoldConfig {
                        max_depth: None,
                        revisit: Revisit::Memoize,
                    },
                ),
                "{children:?} {unmarked:?} root {root}",
            );
            assert_eq!(fixpoint.entered, memoized.entered, "root {root}");
        }
    }
}

/// A monotone fold over a random graph: a node's answer is its own bit, or
/// its successors' bits, shifted along some edges. Some nodes are leaves, and
/// some edges stop the node's remaining successors while their answer is
/// still missing, so a later round can reach a successor the first pass
/// never did.
struct Bits<'a> {
    children: &'a [Vec<usize>],
}

impl Bits<'_> {
    fn is_leaf(node: usize) -> bool {
        node % 7 == 6
    }

    fn own(node: usize) -> u64 {
        1 << (node % 64)
    }

    fn absorb_edge(parent: usize, child: usize, accumulator: &mut u64, value: Option<u64>) -> bool {
        let Some(value) = value else {
            return !(parent + child).is_multiple_of(4);
        };
        *accumulator |= if (parent + child).is_multiple_of(3) {
            value.rotate_left(1)
        } else {
            value
        };
        true
    }
}

impl OpenFold for Bits<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = u64;
    type Accumulator = (usize, u64);

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(&self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<u64, (usize, u64)> {
        if Self::is_leaf(*node) {
            FoldEnter::Leaf(Some(Self::own(*node)))
        } else {
            FoldEnter::Fold {
                accumulator: (*node, Self::own(*node)),
                marks: MarkScope::Shared,
            }
        }
    }

    fn absorb(
        &mut self,
        (parent, accumulator): &mut (usize, u64),
        child: &usize,
        value: Option<u64>,
    ) -> ControlFlow<()> {
        if Self::absorb_edge(*parent, *child, accumulator, value) {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    fn finish(&mut self, _node: &usize, (_, accumulator): (usize, u64)) -> Option<u64> {
        Some(accumulator)
    }
}

/// The least fixpoint of `Bits` on every node, by iterating every node's
/// equation from `None` until nothing changes.
fn kleene(children: &[Vec<usize>]) -> Vec<Option<u64>> {
    let mut values = vec![None; children.len()];
    loop {
        let mut changed = false;
        for node in 0..children.len() {
            let value = if Bits::is_leaf(node) {
                Bits::own(node)
            } else {
                let mut accumulator = Bits::own(node);
                for &child in &children[node] {
                    if !Bits::absorb_edge(node, child, &mut accumulator, values[child]) {
                        break;
                    }
                }
                accumulator
            };
            changed |= values[node] != Some(value);
            values[node] = Some(value);
        }
        if !changed {
            return values;
        }
    }
}

#[test]
fn monotone_folds_reach_the_least_fixpoint_on_random_cyclic_graphs() {
    let mut random = Rng::new(0x00C7_C11C);
    for _ in 0..2_000 {
        let count = 1 + random.below(12);
        let children: Vec<Vec<usize>> = (0..count)
            .map(|_| (0..random.below(5)).map(|_| random.below(count)).collect())
            .collect();
        let least = kleene(&children);
        for (root, &expected) in least.iter().enumerate() {
            let mut bits = Bits {
                children: &children,
            };
            assert_eq!(
                open_fold(&mut bits, root, ITERATE),
                Ok(expected),
                "{children:?} root {root}",
            );
        }
    }
}

#[test]
fn a_cycle_that_a_refold_links_to_an_open_node_joins_its_cycle() {
    // 0 -> 1 -> 0 is found first. 2's first fold stops at its own missing
    // answer before its edge to 1, so 2 looks like a cycle of its own; its
    // re-fold then reaches 1, which is still open. 2 belongs to the cycle of
    // 0 and 1, and the rotation on 2 -> 1 spreads every bit around it.
    let children = vec![vec![1, 2], vec![0, 1, 1], vec![2, 1], vec![3]];
    let least = kleene(&children);
    assert_eq!(least[0], Some(u64::MAX));
    for (root, &expected) in least.iter().enumerate() {
        let mut bits = Bits {
            children: &children,
        };
        assert_eq!(
            open_fold(&mut bits, root, ITERATE),
            Ok(expected),
            "root {root}",
        );
    }
}

/// Unions the nodes' own bits, and stops a node's remaining successors at
/// the first one without an answer.
struct UntilMissing {
    children: &'static [&'static [usize]],
    entered: Vec<usize>,
}

impl OpenFold for UntilMissing {
    type Node = usize;
    type Mark = usize;
    type Value = u64;
    type Accumulator = u64;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<u64, u64> {
        self.entered.push(*node);
        FoldEnter::Fold {
            accumulator: 1 << node,
            marks: MarkScope::Shared,
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut u64,
        _child: &usize,
        value: Option<u64>,
    ) -> ControlFlow<()> {
        match value {
            Some(value) => {
                *accumulator |= value;
                ControlFlow::Continue(())
            }
            None => ControlFlow::Break(()),
        }
    }

    fn finish(&mut self, _node: &usize, accumulator: u64) -> Option<u64> {
        Some(accumulator)
    }
}

#[test]
fn a_member_reached_only_in_a_later_round_joins_the_cycle() {
    // 0 -> [0, 1] and 1 -> [0]. The first pass stops at 0's missing answer
    // and never reaches 1. The first round finds 0's answer, goes on to 1,
    // and 1 joins the cycle, so a second round re-folds both.
    let mut fold = UntilMissing {
        children: &[&[0, 1], &[0]],
        entered: Vec::new(),
    };
    assert_eq!(open_fold(&mut fold, 0, ITERATE), Ok(Some(0b11)),);
    assert_eq!(fold.entered, [0, 0, 1, 0, 1]);
}

/// A fold that flips a boolean it holds on every absorb. A cycle with an odd
/// number of edges flips each answer in every round.
struct Flipper {
    children: &'static [&'static [usize]],
    state: bool,
}

impl OpenFold for Flipper {
    type Node = usize;
    type Mark = usize;
    type Value = bool;
    type Accumulator = bool;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, _node: &usize) -> FoldEnter<bool, bool> {
        FoldEnter::Fold {
            accumulator: false,
            marks: MarkScope::Shared,
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut bool,
        _child: &usize,
        _value: Option<bool>,
    ) -> ControlFlow<()> {
        self.state = !self.state;
        *accumulator = self.state;
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: bool) -> Option<bool> {
        Some(accumulator)
    }
}

#[test]
fn a_fold_that_never_settles_reaches_the_round_limit() {
    let mut flipper = Flipper {
        children: &[&[1], &[2], &[0]],
        state: false,
    };
    assert_eq!(
        open_fold(&mut flipper, 0, round_limit(8)),
        Err(OpenFoldError::RoundLimitExceeded {
            limit: 8,
            rounds: 8
        }),
    );
    let mut flipper = Flipper {
        children: &[&[0]],
        state: false,
    };
    assert_eq!(
        open_fold(&mut flipper, 0, round_limit(0)),
        Err(OpenFoldError::RoundLimitExceeded {
            limit: 0,
            rounds: 0
        }),
        "a self-loop is a cycle too",
    );
}

#[test]
fn an_error_leaves_only_final_answers_in_the_table() {
    // 3 reaches 4, whose answer is final at once, and then the cycle
    // 0 -> 1 -> 2 -> 0, which never settles. The provisional answers of the
    // cycle and of 3 never reach the table.
    let mut flipper = Flipper {
        children: &[&[1], &[2], &[0], &[4, 0], &[]],
        state: false,
    };
    let mut memo = OpenFoldMemo::default();
    assert_eq!(
        open_fold_in(&mut memo, &mut flipper, 3, round_limit(4)),
        Err(OpenFoldError::RoundLimitExceeded {
            limit: 4,
            rounds: 4
        }),
    );
    assert_eq!(memo.answer_count(), 1);
    assert_eq!(memo.get(&4), Some(Some(&false)));
}

/// Counts the nodes below a node, up to a cap; `None` marks a node unmarked.
struct Capped {
    children: &'static [&'static [usize]],
    marks: &'static [Option<usize>],
}

impl OpenFold for Capped {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = usize;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        self.marks[*node]
    }

    fn enter(&mut self, _node: &usize) -> FoldEnter<usize, usize> {
        FoldEnter::Fold {
            accumulator: 1,
            marks: MarkScope::Shared,
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut usize,
        _child: &usize,
        value: Option<usize>,
    ) -> ControlFlow<()> {
        *accumulator = (*accumulator + value.unwrap_or_default()).min(5);
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: usize) -> Option<usize> {
        Some(accumulator)
    }
}

#[test]
fn a_self_loop_iterates_to_its_cap() {
    let mut capped = Capped {
        children: &[&[0]],
        marks: &[Some(0)],
    };
    assert_eq!(open_fold(&mut capped, 0, ITERATE), Ok(Some(5)),);
}

#[test]
fn a_cycle_through_an_unmarked_node_is_an_error() {
    // 0 -> 1 -> 0, where 1 has no mark to hold a provisional answer.
    let mut capped = Capped {
        children: &[&[1], &[0]],
        marks: &[Some(0), None],
    };
    assert_eq!(
        open_fold(&mut capped, 0, ITERATE),
        Err(OpenFoldError::UnmarkedCycle),
    );
    // An unmarked node off the cycle folds on every reach.
    let mut capped = Capped {
        children: &[&[1, 1], &[2], &[]],
        marks: &[Some(0), None, Some(2)],
    };
    assert_eq!(open_fold(&mut capped, 0, ITERATE), Ok(Some(5)),);
}

#[test]
fn the_depth_bound_refuses_deeper_children() {
    let children: &'static [&'static [usize]] = &[&[1], &[2], &[]];
    let marks = &[Some(0), Some(1), Some(2)];
    let mut shallow = Capped { children, marks };
    let bounded = |max_depth| OpenFoldConfig {
        max_depth: Some(max_depth),
        revisit: Revisit::Iterate { max_rounds: None },
    };
    assert_eq!(open_fold(&mut shallow, 0, bounded(1)), Ok(Some(2)));
    let mut exact = Capped { children, marks };
    assert_eq!(open_fold(&mut exact, 0, bounded(2)), Ok(Some(3)));
}
