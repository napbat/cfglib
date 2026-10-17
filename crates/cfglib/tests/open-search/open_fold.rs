//! `open_fold_post_order` with the cycle guard under every `MarkScope`, early exit, the depth
//! bound, and nodes that carry no ordering.

use std::ops::ControlFlow;

use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, open_fold_post_order};

use crate::Lcg;
use crate::reference::{Table, chain, simple_routes};

/// Counts the routes from the root to a sink, folding a node once per route that reaches it.
struct RouteCount<'a> {
    table: &'a Table,
    /// The scope each node chooses when it opens an accumulator.
    scopes: Vec<MarkScope>,
    /// Whether nodes carry a cycle-guard key at all.
    guarded: bool,
    entered: Vec<usize>,
    absorbed: Vec<(usize, Option<u64>)>,
    probed: usize,
}

impl<'a> RouteCount<'a> {
    fn new(table: &'a Table, scope: MarkScope) -> Self {
        Self {
            table,
            scopes: vec![scope; table.node_count()],
            guarded: true,
            entered: Vec::new(),
            absorbed: Vec::new(),
            probed: 0,
        }
    }
}

impl OpenFold for RouteCount<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = u64;
    type Accumulator = u64;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        self.probed += 1;
        out.extend_from_slice(&self.table.successors[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        self.guarded.then_some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<u64, u64> {
        self.entered.push(*node);
        if self.table.successors[*node].is_empty() {
            FoldEnter::Leaf(Some(1))
        } else {
            FoldEnter::Fold {
                accumulator: 0,
                marks: self.scopes[*node],
            }
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut u64,
        child: &usize,
        value: Option<u64>,
    ) -> ControlFlow<()> {
        self.absorbed.push((*child, value));
        *accumulator += value.unwrap_or(0);
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: u64) -> Option<u64> {
        Some(accumulator)
    }
}

/// `0 -> {1, 2}`, `1 -> 3`, `2 -> 3`, with 3 a sink.
fn diamond() -> Table {
    Table {
        successors: vec![vec![1, 2], vec![3], vec![3], vec![]],
    }
}

#[test]
fn per_path_marks_count_the_simple_routes_to_a_sink_like_a_brute_force_enumeration() {
    for seed in 1..=8 {
        let table = Table::random(&mut Lcg(seed), 9, 15);
        let expected = simple_routes(&table, 0, None)
            .into_iter()
            .filter(|route| table.successors[*route.last().expect("non-empty")].is_empty())
            .count();

        let mut fold = RouteCount::new(&table, MarkScope::Isolated);
        let counted = open_fold_post_order(&mut fold, 0, OpenFoldConfig::new());
        let expected = u64::try_from(expected).expect("fits");
        assert_eq!(counted, Some(expected), "seed {seed}");
    }
}

#[test]
fn a_grid_folds_to_a_binomial_per_route_and_to_one_under_shared_marks() {
    const SIDE: usize = 6;
    let node = |x: usize, y: usize| x * (SIDE + 1) + y;
    let mut successors = vec![Vec::new(); (SIDE + 1) * (SIDE + 1)];
    for x in 0..=SIDE {
        for y in 0..=SIDE {
            if x < SIDE {
                successors[node(x, y)].push(node(x + 1, y));
            }
            if y < SIDE {
                successors[node(x, y)].push(node(x, y + 1));
            }
        }
    }
    let table = Table { successors };

    let mut per_route = RouteCount::new(&table, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut per_route, 0, OpenFoldConfig::new()),
        Some(924),
        "C(12, 6)"
    );
    assert_eq!(
        per_route
            .entered
            .iter()
            .filter(|&&entered| entered == node(SIDE, SIDE))
            .count(),
        924
    );

    let mut shared = RouteCount::new(&table, MarkScope::Shared);
    assert_eq!(
        open_fold_post_order(&mut shared, 0, OpenFoldConfig::new()),
        Some(1)
    );
    assert_eq!(shared.entered.len(), 49, "every node is folded once");
}

#[test]
fn shared_and_isolated_marks_differ_exactly_in_whether_the_second_route_is_folded() {
    let table = diamond();

    let mut shared = RouteCount::new(&table, MarkScope::Shared);
    assert_eq!(
        open_fold_post_order(&mut shared, 0, OpenFoldConfig::new()),
        Some(1)
    );
    assert_eq!(
        shared.entered,
        [0, 1, 3, 2],
        "node 3 is refused the second time"
    );
    assert_eq!(
        shared.absorbed,
        [(3, Some(1)), (1, Some(1)), (3, None), (2, Some(0))],
        "a refused child is absorbed as None, once"
    );

    let mut isolated = RouteCount::new(&table, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut isolated, 0, OpenFoldConfig::new()),
        Some(2)
    );
    assert_eq!(isolated.entered, [0, 1, 3, 2, 3]);
    assert_eq!(
        isolated.absorbed,
        [(3, Some(1)), (1, Some(1)), (3, Some(1)), (2, Some(1))]
    );
}

#[test]
fn the_scope_a_parent_chooses_decides_what_its_children_remember() {
    let table = diamond();
    let (shared, isolated) = (MarkScope::Shared, MarkScope::Isolated);
    // The scopes of nodes 0, 1 and 2, and how many routes the diamond then counts.
    let cases = [
        ([shared, shared, shared], 1),
        ([isolated, isolated, isolated], 2),
        ([isolated, shared, shared], 2),
        ([shared, isolated, shared], 2),
        ([shared, shared, isolated], 1),
    ];
    for (scopes, expected) in cases {
        let mut fold = RouteCount::new(&table, shared);
        fold.scopes[..3].copy_from_slice(&scopes);
        let counted = open_fold_post_order(&mut fold, 0, OpenFoldConfig::new());
        assert_eq!(counted, Some(expected), "scopes {scopes:?}");
    }
}

#[test]
fn nodes_without_a_key_are_folded_every_time_and_a_keyed_cycle_is_cut() {
    let cycle = Table {
        successors: vec![vec![1], vec![0]],
    };

    let mut keyed = RouteCount::new(&cycle, MarkScope::Isolated);
    assert_eq!(
        open_fold_post_order(&mut keyed, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(
        keyed.entered,
        [0, 1],
        "the cycle guard refuses the return to 0"
    );

    let mut unkeyed = RouteCount::new(&cycle, MarkScope::Isolated);
    unkeyed.guarded = false;
    let bounded = OpenFoldConfig::new().with_max_depth(4);
    assert_eq!(open_fold_post_order(&mut unkeyed, 0, bounded), Some(0));
    assert_eq!(
        unkeyed.entered,
        [0, 1, 0, 1, 0],
        "only the depth bound ends an unguarded walk"
    );

    let table = diamond();
    let mut diamond_fold = RouteCount::new(&table, MarkScope::Shared);
    diamond_fold.guarded = false;
    assert_eq!(
        open_fold_post_order(&mut diamond_fold, 0, OpenFoldConfig::new()),
        Some(2)
    );
    assert_eq!(
        diamond_fold.entered,
        [0, 1, 3, 2, 3],
        "no key, so shared marks have nothing to remember"
    );
}

/// The longest chain below a node, counting the node itself.
struct Height<'a> {
    table: &'a Table,
}

impl OpenFold for Height<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = usize;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(&self.table.successors[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, _node: &usize) -> FoldEnter<usize, usize> {
        FoldEnter::Fold {
            accumulator: 0,
            marks: MarkScope::Isolated,
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut usize,
        _child: &usize,
        value: Option<usize>,
    ) -> ControlFlow<()> {
        *accumulator = (*accumulator).max(value.unwrap_or(0));
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: usize) -> Option<usize> {
        Some(accumulator + 1)
    }
}

#[test]
fn the_depth_bound_refuses_children_deeper_than_the_given_number_of_levels_below_the_root() {
    let chain = Table {
        successors: vec![vec![1], vec![2], vec![3], vec![4], vec![]],
    };
    let mut fold = Height { table: &chain };
    let height = |fold: &mut Height<'_>, config| open_fold_post_order(fold, 0, config);

    assert_eq!(height(&mut fold, OpenFoldConfig::new()), Some(5));
    assert_eq!(
        height(&mut fold, OpenFoldConfig::new().with_max_depth(4)),
        Some(5),
        "the last node is at depth 4"
    );
    assert_eq!(
        height(&mut fold, OpenFoldConfig::new().with_max_depth(3)),
        Some(4)
    );
    assert_eq!(
        height(&mut fold, OpenFoldConfig::new().with_max_depth(2)),
        Some(3)
    );
    assert_eq!(
        height(&mut fold, OpenFoldConfig::new().with_max_depth(0)),
        Some(1),
        "the root is depth zero"
    );
}

/// Answers with the first declaring node reachable through the children, in order.
struct FirstDeclaration<'a> {
    table: &'a Table,
    declares: Vec<bool>,
    entered: Vec<usize>,
}

impl OpenFold for FirstDeclaration<'_> {
    type Node = usize;
    type Mark = usize;
    type Value = usize;
    type Accumulator = Option<usize>;

    fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
        out.extend_from_slice(&self.table.successors[*node]);
    }

    fn mark(&mut self, node: &usize) -> Option<usize> {
        Some(*node)
    }

    fn enter(&mut self, node: &usize) -> FoldEnter<usize, Option<usize>> {
        self.entered.push(*node);
        if self.declares[*node] {
            FoldEnter::Leaf(Some(*node))
        } else {
            FoldEnter::Fold {
                accumulator: None,
                marks: MarkScope::Isolated,
            }
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut Option<usize>,
        _child: &usize,
        value: Option<usize>,
    ) -> ControlFlow<()> {
        if value.is_some() {
            *accumulator = value;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &usize, accumulator: Option<usize>) -> Option<usize> {
        accumulator
    }
}

#[test]
fn a_break_in_absorb_finishes_the_node_now_and_leaves_its_remaining_children_unfolded() {
    // 0 -> {1, 2, 3}, 1 -> 4 (declares nothing), 2 -> 5, 3 -> 5, and only 5 declares.
    let table = Table {
        successors: vec![vec![1, 2, 3], vec![4], vec![5], vec![5], vec![], vec![]],
    };
    let mut fold = FirstDeclaration {
        table: &table,
        declares: vec![false, false, false, false, false, true],
        entered: Vec::new(),
    };
    let answer = open_fold_post_order(&mut fold, 0, OpenFoldConfig::new());
    assert_eq!(answer, Some(5));
    assert_eq!(
        fold.entered,
        [0, 1, 4, 2, 5],
        "the break at 2 ends 2, and absorbing 2's answer ends 0: node 3 is never entered"
    );
}

/// A node with a computed value and no ordering at all: a segment of the real line.
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    width: f64,
}

/// Counts the leaves of repeated halving down to width one.
struct Halving;

impl OpenFold for Halving {
    type Node = Segment;
    type Mark = u8;
    type Value = u32;
    type Accumulator = u32;

    fn successors(&mut self, node: &Segment, out: &mut Vec<Segment>) {
        let half = node.width / 2.0;
        out.push(Segment {
            start: node.start,
            width: half,
        });
        out.push(Segment {
            start: node.start + half,
            width: half,
        });
    }

    fn mark(&mut self, _node: &Segment) -> Option<u8> {
        None
    }

    fn enter(&mut self, node: &Segment) -> FoldEnter<u32, u32> {
        if node.width <= 1.0 {
            FoldEnter::Leaf(Some(1))
        } else {
            FoldEnter::Fold {
                accumulator: 0,
                marks: MarkScope::Isolated,
            }
        }
    }

    fn absorb(
        &mut self,
        accumulator: &mut u32,
        _child: &Segment,
        value: Option<u32>,
    ) -> ControlFlow<()> {
        *accumulator += value.unwrap_or(0);
        ControlFlow::Continue(())
    }

    fn finish(&mut self, _node: &Segment, accumulator: u32) -> Option<u32> {
        Some(accumulator)
    }
}

#[test]
fn a_fold_over_computed_nodes_needs_no_ordering_on_them() {
    let fold = |width| {
        open_fold_post_order(
            &mut Halving,
            Segment { start: 0.0, width },
            OpenFoldConfig::new(),
        )
    };
    assert_eq!(fold(8.0), Some(8));
    assert_eq!(
        fold(5.0),
        Some(8),
        "5 -> 2.5 -> 1.25 -> 0.625 is three halvings"
    );
    assert_eq!(fold(1.0), Some(1));
}

#[test]
fn a_root_that_answers_as_a_leaf_never_asks_for_its_children() {
    let table = diamond();
    let mut answering = RouteCount::new(&table, MarkScope::Shared);
    // Node 3 is a sink, so as the root it answers directly.
    assert_eq!(
        open_fold_post_order(&mut answering, 3, OpenFoldConfig::new()),
        Some(1)
    );
    assert_eq!(answering.probed, 0);
    assert_eq!(answering.absorbed.len(), 0);

    let mut declaring = FirstDeclaration {
        table: &table,
        declares: vec![true, false, false, false],
        entered: Vec::new(),
    };
    assert_eq!(
        open_fold_post_order(&mut declaring, 0, OpenFoldConfig::new()),
        Some(0)
    );
    assert_eq!(declaring.entered, [0]);
}

#[test]
fn a_fold_over_a_very_deep_space_costs_heap_not_stack() {
    let table = chain(100_000);
    let mut fold = Height { table: &table };
    assert_eq!(
        open_fold_post_order(&mut fold, 0, OpenFoldConfig::new()),
        Some(100_000)
    );
}
