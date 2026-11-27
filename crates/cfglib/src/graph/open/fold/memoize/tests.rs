extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::super::tests::{Digest, Lookup, Scoped, random_dag};
use super::super::{
    FoldEnter, MarkScope, OpenFold, OpenFoldConfig, OpenFoldMemo, Revisit, open_fold, open_fold_in,
};
use crate::test_util::rng::Rng;

/// Memoizes without a depth bound.
const MEMOIZE: OpenFoldConfig = OpenFoldConfig {
    max_depth: None,
    revisit: Revisit::Memoize,
};

#[test]
fn memoized_folds_a_diamond_base_once_and_keeps_both_routes() {
    // The diamond of `isolated_marks_fold_a_diamond_base_once_per_route`: 2
    // absorbs the answer 3 finished with, so 0 still sees two yielding bases.
    let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let owns = &[None, None, None, Some("member")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold(&mut lookup, 0, MEMOIZE),
        Ok(None),
        "both routes reach 0, so the lookup stays ambiguous",
    );
    assert_eq!(lookup.folded, [0, 1, 3, 2], "the shared base folds once");
}

/// `diamonds` stacked diamonds: the top `3 * level` derives `3 * level + 1`
/// and `3 * level + 2`, which both derive the next top. The last top has no
/// bases, and `2^diamonds` routes lead to it.
fn stacked_diamonds(diamonds: usize) -> Vec<Vec<usize>> {
    let mut bases = Vec::with_capacity(3 * diamonds + 1);
    for level in 0..diamonds {
        let top = 3 * level;
        bases.extend([vec![top + 1, top + 2], vec![top + 3], vec![top + 3]]);
    }
    bases.push(Vec::new());
    bases
}

#[test]
fn memoized_folds_the_bottom_of_twenty_diamonds_once() {
    const DIAMONDS: usize = 20;
    let rows = stacked_diamonds(DIAMONDS);
    let bases: Vec<&[usize]> = rows.iter().map(Vec::as_slice).collect();
    let bottom = rows.len() - 1;
    let mut owns = vec![None; rows.len()];
    owns[bottom] = Some("member");
    let folds_of_bottom =
        |lookup: &Lookup<'_>| lookup.folded.iter().filter(|&&node| node == bottom).count();

    let mut memoized = Lookup::new(&bases, &owns, MarkScope::Isolated);
    assert_eq!(open_fold(&mut memoized, 0, MEMOIZE), Ok(None));
    assert_eq!(folds_of_bottom(&memoized), 1);
    assert_eq!(memoized.folded.len(), rows.len(), "every class folds once");

    let mut isolated = Lookup::new(&bases, &owns, MarkScope::Isolated);
    assert_eq!(
        open_fold(&mut isolated, 0, OpenFoldConfig::default()),
        Ok(None),
    );
    assert_eq!(folds_of_bottom(&isolated), 1 << DIAMONDS);
}

#[test]
fn memoized_caches_nothing_between_a_cycle_and_the_node_it_reenters() {
    // 0 derives 1 and 2, which both derive 3; then 3 -> 4 -> 5 -> 3 is a
    // cycle, and 5 finds the member on its side branch 6. 5 refuses 3 while
    // 3 is open, so the answers of 4 and 5 depend on where the walk entered
    // the cycle and stay uncached; 3 and everything above it are cached. 7
    // reaches 0 and then each of 1 to 5 once more.
    let bases: &[&[usize]] = &[
        &[1, 2],
        &[3],
        &[3],
        &[4],
        &[5],
        &[3, 6],
        &[],
        &[0, 1, 2, 3, 4, 5],
    ];
    let owns = &[None, None, None, None, None, None, Some("side"), None];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(open_fold(&mut lookup, 7, MEMOIZE), Ok(None));
    assert_eq!(
        lookup.folded,
        [7, 0, 1, 3, 4, 5, 6, 2, 4, 5],
        "only 4 and 5 fold again; the second time 3 is cached, so they are too",
    );

    // Rooted at 3, the walk enters the cycle at 3.
    let mut memo = OpenFoldMemo::default();
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_in(&mut memo, &mut lookup, 3, MEMOIZE),
        Ok(Some("side")),
    );
    assert_eq!(memo.get(&3), Some(Some(&"side")));
    assert_eq!(memo.get(&4), None);
    assert_eq!(memo.get(&5), None);
    assert_eq!(memo.get(&6), Some(Some(&"side")));

    for root in 0..bases.len() {
        let mut refusing = Lookup::new(bases, owns, MarkScope::Isolated);
        let mut memoizing = Lookup::new(bases, owns, MarkScope::Isolated);
        assert_eq!(
            open_fold(&mut memoizing, root, MEMOIZE),
            open_fold(&mut refusing, root, OpenFoldConfig::default()),
            "root {root}",
        );
    }
}

#[test]
fn memoized_refuses_a_cycle_back_to_a_node_that_is_still_open() {
    // 0 -> 1 -> 0, and 1 -> 2. The mark of 0, open while 1 folds, stops the
    // walk from entering the cycle again. The answer of 1 depends on where
    // the walk entered the cycle, so only 0 and 2 store theirs.
    struct NodeCount {
        entered: Vec<usize>,
    }

    impl OpenFold for NodeCount {
        type Node = usize;
        type Mark = usize;
        type Value = usize;
        type Accumulator = usize;

        fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
            out.extend_from_slice([&[1][..], &[0, 2], &[]][*node]);
        }

        fn mark(&mut self, node: &usize) -> Option<usize> {
            Some(*node)
        }

        fn enter(&mut self, node: &usize) -> FoldEnter<usize, usize> {
            self.entered.push(*node);
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
            *accumulator += value.unwrap_or_default();
            ControlFlow::Continue(())
        }

        fn finish(&mut self, _node: &usize, accumulator: usize) -> Option<usize> {
            Some(accumulator)
        }
    }

    let mut fold = NodeCount {
        entered: Vec::new(),
    };
    let mut memo = OpenFoldMemo::default();
    assert_eq!(open_fold_in(&mut memo, &mut fold, 0, MEMOIZE), Ok(Some(3)));
    assert_eq!(fold.entered, [0, 1, 2]);
    assert_eq!(memo.get(&0), Some(Some(&3)));
    assert_eq!(memo.get(&1), None);
    assert_eq!(memo.get(&2), Some(Some(&1)));
}

#[test]
fn memoized_enters_each_marked_node_once_under_every_scope() {
    // `Scoped` marks every node, so the memoizing walk enters each node once
    // and hands every later reach the stored answer, whatever scope keeps the
    // marks; the refusing walk absorbs `None` there under a shared scope.
    let children: &[&[usize]] = &[&[1, 3, 1, 3], &[2], &[], &[]];
    let leaves = &[false, false, true, true];
    let stops = &[false; 4];
    let mixed = [
        MarkScope::Shared,
        MarkScope::Isolated,
        MarkScope::Shared,
        MarkScope::Shared,
    ];
    for scopes in [[MarkScope::Shared; 4], [MarkScope::Isolated; 4], mixed] {
        let mut fold = Scoped::new(children, leaves, &scopes, stops);
        assert_eq!(open_fold(&mut fold, 0, MEMOIZE), Ok(Some(0)), "{scopes:?}");
        assert_eq!(fold.entered, [0, 1, 2, 3], "{scopes:?}");
        assert_eq!(
            fold.absorbed,
            [(2, true), (1, true), (3, true), (1, true), (3, true)],
            "{scopes:?}",
        );
    }
}

#[test]
fn memoized_stores_no_answer_the_depth_bound_cut() {
    // Under a bound of 1, the depth bound refuses 1's child 2, so 1's answer
    // depends on its depth and is not stored; the leaf 3 is. A shared scope
    // keeps 1 marked and prunes its second reach, and an isolated 1 folds
    // again, as a node whose answer depended on a cycle does.
    let children: &[&[usize]] = &[&[1, 3, 1, 3], &[2], &[], &[]];
    let leaves = &[false, false, true, true];
    let stops = &[false; 4];
    let bounded = OpenFoldConfig {
        max_depth: Some(1),
        revisit: Revisit::Memoize,
    };
    let mut shared = Scoped::new(children, leaves, &[MarkScope::Shared; 4], stops);
    let mut memo = OpenFoldMemo::default();
    assert_eq!(
        open_fold_in(&mut memo, &mut shared, 0, bounded),
        Ok(Some(0))
    );
    assert_eq!(shared.entered, [0, 1, 3]);
    assert_eq!(
        shared.absorbed,
        [(2, false), (1, true), (3, true), (1, false), (3, true)],
    );
    assert_eq!((memo.get(&0), memo.get(&1)), (None, None));
    assert_eq!(memo.get(&3), Some(Some(&3)));

    let isolated_one = [
        MarkScope::Shared,
        MarkScope::Isolated,
        MarkScope::Shared,
        MarkScope::Shared,
    ];
    for scopes in [[MarkScope::Isolated; 4], isolated_one] {
        let mut fold = Scoped::new(children, leaves, &scopes, stops);
        assert_eq!(open_fold(&mut fold, 0, bounded), Ok(Some(0)), "{scopes:?}");
        assert_eq!(fold.entered, [0, 1, 3, 1], "{scopes:?}");
        assert_eq!(
            fold.absorbed,
            [
                (2, false),
                (1, true),
                (3, true),
                (2, false),
                (1, true),
                (3, true)
            ],
            "{scopes:?}",
        );
    }
}

#[test]
fn memoized_prunes_an_answer_it_did_not_store_only_under_a_shared_scope() {
    // 0 reaches 1 twice, and 1 -> 0 closes a cycle, so 1's answer is not
    // stored. A shared scope keeps 1 marked and refuses its second reach; an
    // isolated one releases the mark, and 1 folds again.
    let children: &[&[usize]] = &[&[1, 1], &[0]];
    let leaves = &[false, false];
    let stops = &[false; 2];

    let mut shared = Scoped::new(children, leaves, &[MarkScope::Shared; 2], stops);
    assert_eq!(open_fold(&mut shared, 0, MEMOIZE), Ok(Some(0)));
    assert_eq!(shared.entered, [0, 1]);
    assert_eq!(shared.absorbed, [(0, false), (1, true), (1, false)]);

    let mut isolated = Scoped::new(children, leaves, &[MarkScope::Isolated; 2], stops);
    assert_eq!(open_fold(&mut isolated, 0, MEMOIZE), Ok(Some(0)));
    assert_eq!(isolated.entered, [0, 1, 1]);
    assert_eq!(
        isolated.absorbed,
        [(0, false), (1, true), (0, false), (1, true)]
    );
}

#[test]
fn memoized_respects_the_depth_bound() {
    let bases: &[&[usize]] = &[&[1], &[2], &[]];
    let owns = &[None, None, Some("deep")];
    let bounded = |max_depth| OpenFoldConfig {
        max_depth: Some(max_depth),
        revisit: Revisit::Memoize,
    };
    let mut shallow = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(open_fold(&mut shallow, 0, bounded(1)), Ok(None));
    let mut exact = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(open_fold(&mut exact, 0, bounded(2)), Ok(Some("deep")));
}

#[test]
fn memoized_reuses_only_whole_answers_and_at_any_depth() {
    // 0 derives 1 and 2, 1 derives 2, and 2 derives the leaf 3. Under a bound
    // of 2, the route 0 -> 1 -> 2 cuts 3 off, so neither 2 nor 1 is stored,
    // and the route 0 -> 2 folds 2 again and reaches 3 within the bound: the
    // walk answers as the refusing one does. Its whole answer for 2 is
    // stored, and a later call reuses it at a depth where a fresh fold under
    // its own bound would cut 3.
    let bases: &[&[usize]] = &[&[1, 2], &[2], &[3], &[]];
    let owns = &[None, None, None, Some("deep")];
    let bounded = |max_depth, revisit| OpenFoldConfig {
        max_depth: Some(max_depth),
        revisit,
    };
    let mut memo = OpenFoldMemo::default();
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_in(&mut memo, &mut lookup, 0, bounded(2, Revisit::Memoize)),
        Ok(Some("deep")),
    );
    assert_eq!(lookup.folded, [0, 1, 2, 2, 3]);
    assert_eq!((memo.get(&0), memo.get(&1)), (None, None));
    assert_eq!(memo.get(&2), Some(Some(&"deep")));
    let mut refusing = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold(&mut refusing, 0, bounded(2, Revisit::Refuse)),
        Ok(Some("deep")),
    );

    assert_eq!(
        open_fold_in(&mut memo, &mut lookup, 1, bounded(1, Revisit::Memoize)),
        Ok(Some("deep")),
    );
    assert_eq!(lookup.folded, [0, 1, 2, 2, 3, 1], "2 is reused, not folded");
}

/// A [`Lookup`] whose listed nodes keep their answers to their routes.
struct RouteBound<'a> {
    lookup: Lookup<'a>,
    route_bound: &'a [usize],
}

impl OpenFold for RouteBound<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = &'static str;
    type Accumulator = super::super::tests::Combined;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        self.lookup.successors(node, out);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        self.lookup.mark(node)
    }

    fn memoize(&mut self, node: &usize) -> bool {
        !self.route_bound.contains(node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<&'static str, Self::Accumulator> {
        self.lookup.enter(node)
    }

    fn absorb(
        &mut self,
        accumulator: &mut Self::Accumulator,
        child: &usize,
        value: Option<&'static str>,
    ) -> ControlFlow<()> {
        self.lookup.absorb(accumulator, child, value)
    }

    fn finish(&mut self, node: &usize, accumulator: Self::Accumulator) -> Option<&'static str> {
        self.lookup.finish(node, accumulator)
    }
}

#[test]
fn memoized_folds_a_route_bound_node_on_every_reach_and_still_guards_it() {
    // The diamond 0 -> 1, 2 -> 3: 3 keeps its answer to its route, so both
    // routes fold it, and the nodes above it are stored as usual.
    let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let owns = &[None, None, None, Some("member")];
    let mut fold = RouteBound {
        lookup: Lookup::new(bases, owns, MarkScope::Isolated),
        route_bound: &[3],
    };
    let mut memo = OpenFoldMemo::default();
    assert_eq!(open_fold_in(&mut memo, &mut fold, 0, MEMOIZE), Ok(None));
    assert_eq!(fold.lookup.folded, [0, 1, 3, 2, 3]);
    assert_eq!(memo.get(&3), None);
    assert_eq!(memo.get(&1), Some(Some(&"member")));

    // 1 -> 0 -> 1: the mark of the route-bound 1 still refuses the cycle.
    let cycle: &[&[usize]] = &[&[1], &[0]];
    let mut fold = RouteBound {
        lookup: Lookup::new(cycle, &[None, None], MarkScope::Isolated),
        route_bound: &[1],
    };
    let mut memo = OpenFoldMemo::default();
    assert_eq!(open_fold_in(&mut memo, &mut fold, 1, MEMOIZE), Ok(None));
    assert_eq!(fold.lookup.folded, [1, 0]);
    assert!(memo.is_empty(), "0 depends on the cycle and 1 on its route");
}

#[test]
fn memoized_reuses_the_answers_of_earlier_calls_until_they_are_removed() {
    // The diamond 0 -> 1, 2 -> 3, folded from 1 and then from 0 with one
    // table: the second call folds only 0 and 2.
    let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let owns = &[None, None, None, Some("member")];
    let mut memo = OpenFoldMemo::default();
    assert!(memo.is_empty());
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_in(&mut memo, &mut lookup, 1, MEMOIZE),
        Ok(Some("member")),
    );
    assert_eq!(
        open_fold_in(&mut memo, &mut lookup, 0, MEMOIZE),
        Ok(None),
        "both routes still reach 0",
    );
    assert_eq!(lookup.folded, [1, 3, 0, 2]);
    assert_eq!(memo.answer_count(), 4);
    assert_eq!(memo.get(&0), Some(None));

    // A removed answer is folded again; the answers it fed stay stored.
    assert_eq!(memo.remove(&2), Some(Some("member")));
    assert_eq!(memo.remove(&2), None);
    assert_eq!(open_fold_in(&mut memo, &mut lookup, 0, MEMOIZE), Ok(None),);
    assert_eq!(
        open_fold_in(&mut memo, &mut lookup, 2, MEMOIZE),
        Ok(Some("member")),
    );
    assert_eq!(lookup.folded, [1, 3, 0, 2, 2]);

    memo.clear();
    assert!(memo.is_empty());
    assert_eq!(memo.get(&3), None);
}

#[test]
fn memoized_and_isolated_folds_agree_on_random_dags() {
    let mut random = Rng::new(0x63_5EED);
    for _ in 0..500 {
        let (children, unmarked) = random_dag(&mut random);
        for root in 0..children.len() {
            let mut isolated = Digest {
                children: &children,
                unmarked: &unmarked,
                entered: Vec::new(),
            };
            let mut memoized = Digest {
                children: &children,
                unmarked: &unmarked,
                entered: Vec::new(),
            };
            assert_eq!(
                open_fold(&mut memoized, root, MEMOIZE),
                open_fold(&mut isolated, root, OpenFoldConfig::default()),
                "{children:?} {unmarked:?} root {root}",
            );
            let mut seen = vec![false; children.len()];
            for &node in &memoized.entered {
                if !unmarked[node] {
                    assert!(!seen[node], "a marked node folds once");
                    seen[node] = true;
                }
            }
        }
    }
}

#[test]
fn one_table_across_every_root_agrees_with_fresh_tables_on_random_dags() {
    let mut random = Rng::new(0x7A_B1E5);
    for _ in 0..500 {
        let (children, unmarked) = random_dag(&mut random);
        let mut memo = OpenFoldMemo::default();
        let mut shared = Digest {
            children: &children,
            unmarked: &unmarked,
            entered: Vec::new(),
        };
        for root in (0..children.len()).rev() {
            let mut fresh = Digest {
                children: &children,
                unmarked: &unmarked,
                entered: Vec::new(),
            };
            assert_eq!(
                open_fold_in(&mut memo, &mut shared, root, MEMOIZE),
                open_fold(&mut fresh, root, MEMOIZE),
                "{children:?} {unmarked:?} root {root}",
            );
        }
        let mut seen = vec![false; children.len()];
        for &node in &shared.entered {
            if !unmarked[node] {
                assert!(!seen[node], "a marked node folds once over every root");
                seen[node] = true;
            }
        }
    }
}
