extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::{
    FoldEnter, MarkScope, OpenFold, OpenFoldConfig, open_fold_post_order,
    open_fold_post_order_memoized,
};
use crate::test_util::rng::Rng;

/// A member-lookup fold over a static base table: a node with an `own`
/// answer is a leaf, an unanswered node combines its bases with sticky
/// ambiguity, and every node folds with the given mark scope. Every node is
/// its own mark and memo key; only the memoized driver reads the key.
struct Lookup<'a> {
    bases: &'a [&'a [usize]],
    owns: &'a [Option<&'static str>],
    scope: MarkScope,
    folded: Vec<usize>,
}

impl<'a> Lookup<'a> {
    fn new(bases: &'a [&'a [usize]], owns: &'a [Option<&'static str>], scope: MarkScope) -> Self {
        Self {
            bases,
            owns,
            scope,
            folded: Vec::new(),
        }
    }
}

#[derive(Debug, PartialEq)]
enum Combined {
    Nothing,
    One(&'static str),
    Ambiguous,
}

impl OpenFold for Lookup<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = &'static str;
    type Accumulator = Combined;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.bases[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn memo_key(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<&'static str, Combined> {
        self.folded.push(*node);
        match self.owns[*node] {
            Some(name) => FoldEnter::Leaf(Some(name)),
            None => FoldEnter::Fold {
                accumulator: Combined::Nothing,
                marks: self.scope,
            },
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut Combined,
        _child: &usize,
        value: Option<&'static str>,
    ) -> ControlFlow<()> {
        if let Some(value) = value {
            *accumulator = match accumulator {
                Combined::Nothing => Combined::One(value),
                Combined::One(_) | Combined::Ambiguous => Combined::Ambiguous,
            };
        }
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: Combined) -> Option<&'static str> {
        match accumulator {
            Combined::One(value) => Some(value),
            Combined::Nothing | Combined::Ambiguous => None,
        }
    }
}

#[test]
fn a_chain_propagates_the_deep_answer() {
    let bases: &[&[usize]] = &[&[1], &[2], &[]];
    let owns = &[None, None, Some("deep")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
        Some("deep"),
    );
}

#[test]
fn isolated_marks_fold_a_diamond_base_once_per_route() {
    // 0 derives 1 and 2; both derive 3, which declares the member. Per-path
    // marks fold 3 twice, so 0 sees two yielding bases: ambiguous.
    let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let owns = &[None, None, None, Some("member")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
        None,
    );
    assert_eq!(
        lookup.folded,
        [0, 1, 3, 2, 3],
        "the shared base folds once per route"
    );
}

#[test]
fn an_own_declaration_hides_the_ambiguous_bases() {
    // 1 declares the member itself, so its diamond below never folds and 0
    // receives exactly one yielding base.
    let bases: &[&[usize]] = &[&[1], &[2, 3], &[4], &[4], &[]];
    let owns = &[None, Some("hidden"), None, None, Some("member")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
        Some("hidden"),
    );
    assert_eq!(lookup.folded, [0, 1], "the leaf's bases are never visited");
}

#[test]
fn shared_marks_refuse_the_second_route() {
    // The same diamond under persistent marks: 3 folds once, the second
    // route absorbs `None`, and the lookup resolves unambiguously.
    let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let owns = &[None, None, None, Some("member")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Shared);
    assert_eq!(
        open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
        Some("member"),
    );
    assert_eq!(
        lookup.folded,
        [0, 1, 3, 2],
        "the marked base never re-folds"
    );
}

#[test]
fn cycles_contribute_nothing_on_their_route() {
    // 0 -> 1 -> 0 with the answer on a side branch of 1.
    let bases: &[&[usize]] = &[&[1], &[0, 2], &[]];
    let owns = &[None, None, Some("side")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
        Some("side"),
    );
}

#[test]
fn the_depth_bound_refuses_deeper_children() {
    let bases: &[&[usize]] = &[&[1], &[2], &[]];
    let owns = &[None, None, Some("deep")];
    let mut shallow = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut shallow, 0, OpenFoldConfig::new().with_max_depth(1)),
        None,
    );
    let mut exact = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut exact, 0, OpenFoldConfig::new().with_max_depth(2)),
        Some("deep"),
    );
}

/// A first-match fold: the first yielding child wins and stops its
/// siblings, mirroring a base-chain lookup that returns the first hit.
struct FirstMatch<'a> {
    children: &'a [&'a [usize]],
    owns: &'a [Option<&'static str>],
    folded: Vec<usize>,
}

impl OpenFold for FirstMatch<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = &'static str;
    type Accumulator = Option<&'static str>;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<&'static str, Option<&'static str>> {
        self.folded.push(*node);
        match self.owns[*node] {
            Some(name) => FoldEnter::Leaf(Some(name)),
            None => FoldEnter::Fold {
                accumulator: None,
                marks: MarkScope::Shared,
            },
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut Option<&'static str>,
        _child: &usize,
        value: Option<&'static str>,
    ) -> ControlFlow<()> {
        if value.is_some() {
            *accumulator = value;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: Option<&'static str>) -> Option<&'static str> {
        accumulator
    }
}

#[test]
fn a_breaking_absorb_skips_the_remaining_children() {
    let children: &[&[usize]] = &[&[1, 2, 3], &[], &[], &[]];
    let owns = &[None, Some("first"), Some("second"), Some("third")];
    let mut lookup = FirstMatch {
        children,
        owns,
        folded: Vec::new(),
    };
    assert_eq!(
        open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
        Some("first"),
    );
    assert_eq!(lookup.folded, [0, 1], "later siblings are never entered");
}

/// A fold whose nodes carry no marks at all: revisits are allowed, so a
/// shared subterm is evaluated once per reference, like a value algebra.
struct Unmarked<'a> {
    children: &'a [&'a [usize]],
    folded: Vec<usize>,
}

impl OpenFold for Unmarked<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = usize;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.children[*node]);
    }

    fn mark(&mut self, _node: &usize) -> Option<usize> {
        None
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<usize, usize> {
        self.folded.push(*node);
        if self.children[*node].is_empty() {
            FoldEnter::Leaf(Some(1))
        } else {
            FoldEnter::Fold {
                accumulator: 0,
                marks: MarkScope::Shared,
            }
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

#[test]
fn unmarked_nodes_fold_on_every_reach() {
    // Both 1 and 2 share leaf 3; without marks it counts once per route.
    let children: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let mut fold = Unmarked {
        children,
        folded: Vec::new(),
    };
    assert_eq!(
        open_fold_post_order(&mut fold, 0, OpenFoldConfig::new()),
        Some(2),
    );
    assert_eq!(fold.folded, [0, 1, 3, 2, 3]);
}

/// A fold whose nodes each choose their own mark scope and say whether they
/// stop after their first child, recording what each node absorbed. Every
/// node folds to itself, so a refused child shows as an absorbed `None`.
struct Scoped<'a> {
    children: &'a [&'a [usize]],
    leaves: &'a [bool],
    scopes: &'a [MarkScope],
    stops: &'a [bool],
    entered: Vec<usize>,
    absorbed: Vec<(usize, bool)>,
}

impl<'a> Scoped<'a> {
    fn new(
        children: &'a [&'a [usize]],
        leaves: &'a [bool],
        scopes: &'a [MarkScope],
        stops: &'a [bool],
    ) -> Self {
        Self {
            children,
            leaves,
            scopes,
            stops,
            entered: Vec::new(),
            absorbed: Vec::new(),
        }
    }
}

impl OpenFold for Scoped<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = usize;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<usize, usize> {
        self.entered.push(*node);
        if self.leaves[*node] {
            FoldEnter::Leaf(Some(*node))
        } else {
            FoldEnter::Fold {
                accumulator: *node,
                marks: self.scopes[*node],
            }
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut usize,
        child: &usize,
        value: Option<usize>,
    ) -> ControlFlow<()> {
        self.absorbed.push((*child, value.is_some()));
        if self.stops[*accumulator] {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn finish(&mut self, _node: &usize, accumulator: usize) -> Option<usize> {
        Some(accumulator)
    }
}

#[test]
fn an_isolated_parent_forgets_a_leafs_mark_between_children() {
    // 0 reaches the leaf 1 twice. Persistent marks refuse the second reach; an
    // isolated parent starts each child from its own entry marks, so the leaf
    // answers both times.
    let children: &[&[usize]] = &[&[1, 1], &[]];
    let leaves = &[false, true];
    let stops = &[false, false];

    let mut isolated = Scoped::new(children, leaves, &[MarkScope::Isolated; 2], stops);
    assert_eq!(
        open_fold_post_order(&mut isolated, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(isolated.absorbed, [(1, true), (1, true)]);

    let mut shared = Scoped::new(children, leaves, &[MarkScope::Shared; 2], stops);
    assert_eq!(
        open_fold_post_order(&mut shared, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(shared.absorbed, [(1, true), (1, false)]);
}

#[test]
fn an_isolated_ancestor_takes_back_what_a_shared_subtree_marked() {
    // 0 is isolated; 1 and 2 are shared and both fold the leaf 3. What 1's
    // subtree marked (its own key and 3) is gone before 2 starts, so 2's
    // subtree folds 3 too instead of refusing it.
    let children: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let leaves = &[false, false, false, true];
    let scopes = &[
        MarkScope::Isolated,
        MarkScope::Shared,
        MarkScope::Shared,
        MarkScope::Shared,
    ];
    let mut fold = Scoped::new(children, leaves, scopes, &[false; 4]);
    assert_eq!(
        open_fold_post_order(&mut fold, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(fold.entered, [0, 1, 3, 2, 3]);
    assert_eq!(fold.absorbed, [(3, true), (1, true), (3, true), (2, true)]);
}

#[test]
fn an_isolated_node_takes_back_its_own_key_and_its_subtrees_when_it_finishes() {
    // 0 is shared and reaches 1 twice around the leaf 3. 1 is isolated: its
    // subtree marks 2, and once 1 finishes that mark and 1's own key are gone,
    // so the second reach of 1 folds it (and 2) again. Marks 0 itself set
    // before — 1's first key, then the leaf 3 — stay.
    let children: &[&[usize]] = &[&[1, 3, 1, 3], &[2], &[], &[]];
    let leaves = &[false, false, true, true];
    let scopes = &[
        MarkScope::Shared,
        MarkScope::Isolated,
        MarkScope::Shared,
        MarkScope::Shared,
    ];
    let mut fold = Scoped::new(children, leaves, scopes, &[false; 4]);
    assert_eq!(
        open_fold_post_order(&mut fold, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(fold.entered, [0, 1, 2, 3, 1, 2]);
    assert_eq!(
        fold.absorbed,
        [
            (2, true),
            (1, true),
            (3, true),
            (2, true),
            (1, true),
            (3, false)
        ]
    );
}

#[test]
fn a_stopped_node_leaves_its_children_to_no_one_else() {
    // 1 stops after its first child, 2, leaving 3 unfolded. When 1 finishes
    // the walk resumes with 0's next child, 4 — not with 1's leftover 3.
    let children: &[&[usize]] = &[&[1, 4], &[2, 3], &[], &[], &[]];
    let leaves = &[false, false, true, true, true];
    let scopes = &[MarkScope::Shared; 5];
    let stops = &[false, true, false, false, false];
    let mut fold = Scoped::new(children, leaves, scopes, stops);
    assert_eq!(
        open_fold_post_order(&mut fold, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(fold.entered, [0, 1, 2, 4]);
    assert_eq!(fold.absorbed, [(2, true), (1, true), (4, true)]);
}

/// Nodes `0..count`, each deriving the next two: a diamond at every step, so
/// the member at the last node is reached along a Fibonacci number of routes.
fn skip_chain(count: usize) -> Vec<Vec<usize>> {
    (0..count)
        .map(|node| (node + 1..count).take(2).collect())
        .collect()
}

#[test]
fn marks_keep_their_meaning_beyond_a_handful_of_nodes() {
    // Twelve nodes mark more keys than a small walk holds inline: the shared
    // scope must still refuse every second route, and the isolated scope must
    // still release a node's key when it finishes, so a route can fold it again.
    const COUNT: usize = 12;
    let rows = skip_chain(COUNT);
    let bases: Vec<&[usize]> = rows.iter().map(Vec::as_slice).collect();
    let mut owns: Vec<Option<&'static str>> = (0..COUNT).map(|_| None).collect();
    owns[COUNT - 1] = Some("member");

    let mut shared = Lookup::new(&bases, &owns, MarkScope::Shared);
    assert_eq!(
        open_fold_post_order(&mut shared, 0, OpenFoldConfig::new()),
        Some("member"),
    );
    assert_eq!(shared.folded, (0..COUNT).collect::<Vec<_>>());

    let mut isolated = Lookup::new(&bases, &owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut isolated, 0, OpenFoldConfig::new()),
        None,
        "two routes reach the member, so the answer is ambiguous",
    );
    let mut folds = [0_usize; COUNT];
    for &node in &isolated.folded {
        folds[node] += 1;
    }
    let mut routes = [1_usize; COUNT];
    for node in 2..COUNT {
        routes[node] = routes[node - 1] + routes[node - 2];
    }
    assert_eq!(folds, routes, "each node folds once per route to it");
}

/// Folds `root` under both drivers with fresh lookups and returns the plain
/// driver's answer and the memoized one's.
fn plain_and_memoized(
    bases: &[&[usize]],
    owns: &[Option<&'static str>],
    root: usize,
) -> (Option<&'static str>, Option<&'static str>) {
    let mut plain = Lookup::new(bases, owns, MarkScope::Isolated);
    let mut memoized = Lookup::new(bases, owns, MarkScope::Isolated);
    (
        open_fold_post_order(&mut plain, root, OpenFoldConfig::new()),
        open_fold_post_order_memoized(&mut memoized, root, OpenFoldConfig::new()),
    )
}

#[test]
fn memoized_folds_a_diamond_base_once_and_keeps_both_routes() {
    // The diamond of `isolated_marks_fold_a_diamond_base_once_per_route`: 2
    // absorbs the answer 3 finished with, so 0 still sees two yielding bases.
    let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let owns = &[None, None, None, Some("member")];
    let mut lookup = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order_memoized(&mut lookup, 0, OpenFoldConfig::new()),
        None,
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
    assert_eq!(
        open_fold_post_order_memoized(&mut memoized, 0, OpenFoldConfig::new()),
        None,
    );
    assert_eq!(folds_of_bottom(&memoized), 1);
    assert_eq!(memoized.folded.len(), rows.len(), "every class folds once");

    let mut isolated = Lookup::new(&bases, &owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut isolated, 0, OpenFoldConfig::new()),
        None,
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
    open_fold_post_order_memoized(&mut lookup, 7, OpenFoldConfig::new());
    assert_eq!(
        lookup.folded,
        [7, 0, 1, 3, 4, 5, 6, 2, 4, 5],
        "only 4 and 5 fold again; the second time 3 is cached, so they are too",
    );

    for root in 0..bases.len() {
        let (plain, memoized) = plain_and_memoized(bases, owns, root);
        assert_eq!(memoized, plain, "root {root}");
    }
}

#[test]
fn memoized_refuses_a_cycle_through_a_node_with_a_memo_key_but_no_mark() {
    // 0 -> 1 -> 0, and 1 -> 2. No node has a mark, so only the memo key of
    // 0 stops the walk from entering the cycle again.
    struct KeysOnly {
        entered: Vec<usize>,
    }

    impl OpenFold for KeysOnly {
        type Node = usize;
        type Mark = usize;
        type Value = usize;
        type Accumulator = usize;

        fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
            out.extend_from_slice([&[1][..], &[0, 2], &[]][*node]);
        }

        fn mark(&mut self, _node: &usize) -> Option<usize> {
            None
        }

        fn memo_key(&mut self, node: &usize) -> Option<usize> {
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

    let mut fold = KeysOnly {
        entered: Vec::new(),
    };
    assert_eq!(
        open_fold_post_order_memoized(&mut fold, 0, OpenFoldConfig::new()),
        Some(3),
    );
    assert_eq!(fold.entered, [0, 1, 2]);
}

#[test]
fn without_memo_keys_the_memoized_walk_is_the_plain_walk() {
    // `Scoped` keeps the default `memo_key`, so the memoized driver must
    // enter and absorb exactly what the plain one does, under every scope.
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
        for config in [
            OpenFoldConfig::new(),
            OpenFoldConfig::new().with_max_depth(1),
        ] {
            let mut plain = Scoped::new(children, leaves, &scopes, stops);
            let mut memoized = Scoped::new(children, leaves, &scopes, stops);
            assert_eq!(
                open_fold_post_order_memoized(&mut memoized, 0, config),
                open_fold_post_order(&mut plain, 0, config),
            );
            assert_eq!(memoized.entered, plain.entered);
            assert_eq!(memoized.absorbed, plain.absorbed);
        }
    }
}

#[test]
fn memoized_respects_the_depth_bound() {
    let bases: &[&[usize]] = &[&[1], &[2], &[]];
    let owns = &[None, None, Some("deep")];
    let mut shallow = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order_memoized(&mut shallow, 0, OpenFoldConfig::new().with_max_depth(1)),
        None,
    );
    let mut exact = Lookup::new(bases, owns, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order_memoized(&mut exact, 0, OpenFoldConfig::new().with_max_depth(2)),
        Some("deep"),
    );
}

/// A fold whose answer is a hash of the node and of its children's answers in
/// order. Some nodes answer as leaves, some stop after a child, some have no
/// mark (and so no memo key), and some answer nothing. On a graph without
/// cycles every answer is a function of the node alone, whatever the route.
struct Digest<'a> {
    children: &'a [Vec<usize>],
    unmarked: &'a [bool],
    entered: Vec<usize>,
}

impl OpenFold for Digest<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = u64;
    type Accumulator = u64;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(&self.children[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        (!self.unmarked[*node]).then_some(*node)
    }

    fn memo_key(&mut self, node: &usize) -> Option<usize> {
        self.mark(node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<u64, u64> {
        self.entered.push(*node);
        let seed = u64::try_from(*node).expect("a test node fits in u64") + 1;
        if node % 5 == 4 {
            FoldEnter::Leaf(Some(seed))
        } else {
            FoldEnter::Fold {
                accumulator: seed,
                marks: MarkScope::Isolated,
            }
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut u64,
        _child: &usize,
        value: Option<u64>,
    ) -> ControlFlow<()> {
        *accumulator = accumulator
            .wrapping_mul(31)
            .wrapping_add(value.map_or(7, |value| value.wrapping_add(1)));
        if accumulator.is_multiple_of(13) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn finish(&mut self, _node: &usize, accumulator: u64) -> Option<u64> {
        (!accumulator.is_multiple_of(5)).then_some(accumulator)
    }
}

/// A random graph of up to twelve nodes whose edges all point to a later
/// node, with parallel edges, and the nodes that have no mark.
fn random_dag(random: &mut Rng) -> (Vec<Vec<usize>>, Vec<bool>) {
    let count = 1 + random.below(12);
    let children = (0..count)
        .map(|node| {
            let later = count - node - 1;
            if later == 0 {
                return Vec::new();
            }
            (0..random.below(4))
                .map(|_| node + 1 + random.below(later))
                .collect()
        })
        .collect();
    let unmarked = (0..count).map(|_| random.one_in(5)).collect();
    (children, unmarked)
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
                open_fold_post_order_memoized(&mut memoized, root, OpenFoldConfig::new()),
                open_fold_post_order(&mut isolated, root, OpenFoldConfig::new()),
                "{children:?} {unmarked:?} root {root}",
            );
            let mut seen = vec![false; children.len()];
            for &node in &memoized.entered {
                if !unmarked[node] {
                    assert!(!seen[node], "a node with a memo key folds once");
                    seen[node] = true;
                }
            }
        }
    }
}
