use cfglib::{
    Cfg, Direction, DominatorTree, Edge, EdgeId, EdgeKind, EdgeView, FilteredEdges, KeyedGraph,
    NodeId, RootedView, TraversalDirection, TryEdgeProblem, TrySolveError, breadth_first_edges,
    remove_empty_blocks_mapped, split_node_at_points, try_solve_edge_problem_from,
    verify_edge_view,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Normal,
    SwitchDefault,
    SwitchCase(i32),
    Handler { order: u8 },
    Unwind,
    Continuation { call_site: u32, ordinal: u8 },
    Synthetic,
}

fn is_normal(_: EdgeId, edge: &Edge<Route>) -> bool {
    matches!(
        edge.payload(),
        Route::Normal
            | Route::SwitchDefault
            | Route::SwitchCase(_)
            | Route::Continuation { .. }
            | Route::Synthetic
    )
}

#[test]
fn normal_and_full_flow_algorithms_share_storage_but_not_reachability() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let then_block = cfg.new_block();
    let else_block = cfg.new_block();
    let merge = cfg.new_block();
    let handler = cfg.new_block();
    let entry = cfg.entry();
    cfg.add_edge_with_payload(entry, then_block, EdgeKind::ConditionalTrue, Route::Normal);
    cfg.add_edge_with_payload(entry, else_block, EdgeKind::ConditionalFalse, Route::Normal);
    cfg.add_edge_with_payload(then_block, merge, EdgeKind::Fallthrough, Route::Normal);
    cfg.add_edge_with_payload(else_block, merge, EdgeKind::Fallthrough, Route::Normal);
    let exceptional = cfg.add_edge_with_payload(
        entry,
        handler,
        EdgeKind::ExceptionHandler,
        Route::Handler { order: 0 },
    );

    let full = DominatorTree::compute(&cfg);
    assert!(full.is_reachable(handler));

    let normal = FilteredEdges::new(&cfg, is_normal);
    assert!(verify_edge_view(&normal).is_ok());
    assert_eq!(normal.root(), entry);
    assert_eq!(normal.edge_bound(), cfg.edge_bound());
    assert!(!normal.edge_ids().any(|edge| edge == exceptional));
    let normal_dominators = DominatorTree::compute(&normal);
    assert!(normal_dominators.dominates(entry, merge));
    assert!(!normal_dominators.is_reachable(handler));
}

#[test]
fn parallel_switch_and_continuation_edges_keep_identity_and_order() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let target = cfg.new_block();
    let resume = cfg.new_block();
    let entry = cfg.entry();
    let default =
        cfg.add_edge_with_payload(entry, target, EdgeKind::SwitchCase, Route::SwitchDefault);
    let case = cfg.add_edge_with_payload(entry, target, EdgeKind::SwitchCase, Route::SwitchCase(7));
    let first_continuation = cfg.add_edge_with_payload(
        target,
        resume,
        EdgeKind::CallReturn,
        Route::Continuation {
            call_site: 42,
            ordinal: 0,
        },
    );
    let second_continuation = cfg.add_edge_with_payload(
        target,
        resume,
        EdgeKind::CallReturn,
        Route::Continuation {
            call_site: 42,
            ordinal: 1,
        },
    );

    assert_ne!(default, case);
    assert_ne!(first_continuation, second_continuation);
    assert_eq!(cfg.outgoing(entry).collect::<Vec<_>>(), [default, case]);
    assert_eq!(
        cfg.outgoing(target).collect::<Vec<_>>(),
        [first_continuation, second_continuation]
    );

    let steps = breadth_first_edges(&cfg, entry, TraversalDirection::Outgoing);
    let ids: Vec<_> = steps.iter().map(|step| step.edge).collect();
    assert_eq!(
        ids,
        [default, case, first_continuation, second_continuation]
    );
    assert_eq!(ids[0].index() + 1, ids[1].index());
}

#[test]
fn split_redirect_bypass_and_clone_report_metadata_preserving_mappings() {
    let mut cfg = Cfg::<u8, Route>::with_edge_payload();
    let exit = cfg.new_block();
    let entry = cfg.entry();
    cfg.block_mut(entry).instructions_mut().extend([10, 20, 30]);
    let outgoing = cfg.add_edge_with_payload(entry, exit, EdgeKind::Jump, Route::SwitchCase(9));

    let (parts, split) = split_node_at_points(
        &mut cfg,
        entry,
        [(1, Route::Synthetic), (2, Route::Synthetic)],
    )
    .unwrap();
    assert_eq!(split.blocks(entry), Some(parts.as_slice()));
    assert_eq!(split.edges(outgoing), Some([outgoing].as_slice()));
    assert_eq!(cfg.edge(outgoing).source(), parts[2]);
    assert_eq!(cfg.edge(outgoing).payload(), &Route::SwitchCase(9));

    let mut clone_blocks = parts.clone();
    clone_blocks.push(exit);
    let (clone, cloned) = cfg.subgraph_mapped(&clone_blocks);
    let cloned_outgoing = cloned.edges(outgoing).unwrap()[0];
    assert_eq!(clone.edge(cloned_outgoing).payload(), &Route::SwitchCase(9));
    assert!(cloned.created_edges().contains(&cloned_outgoing));

    let mut bypass = Cfg::<(), Route>::with_edge_payload();
    let empty = bypass.new_block();
    let target = bypass.new_block();
    let entry = bypass.entry();
    let incoming = bypass.add_edge_with_payload(entry, empty, EdgeKind::Jump, Route::SwitchCase(3));
    let removed =
        bypass.add_edge_with_payload(empty, target, EdgeKind::Fallthrough, Route::Synthetic);
    let (count, mapping) = remove_empty_blocks_mapped(&mut bypass);
    assert_eq!(count, 1);
    assert_eq!(mapping.edges(incoming), Some([incoming].as_slice()));
    assert_eq!(mapping.edges(removed), Some([].as_slice()));
    assert_eq!(mapping.blocks(empty), Some([].as_slice()));
    assert_eq!(bypass.edge(incoming).target(), target);
    assert_eq!(bypass.edge(incoming).payload(), &Route::SwitchCase(3));
}

#[test]
fn handler_and_unwind_payloads_remain_distinct_and_ordered() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let first = cfg.new_block();
    let second = cfg.new_block();
    let unwind = cfg.new_block();
    let entry = cfg.entry();
    let edges = [
        cfg.add_edge_with_payload(
            entry,
            first,
            EdgeKind::ExceptionHandler,
            Route::Handler { order: 0 },
        ),
        cfg.add_edge_with_payload(
            entry,
            second,
            EdgeKind::ExceptionHandler,
            Route::Handler { order: 1 },
        ),
        cfg.add_edge_with_payload(entry, unwind, EdgeKind::ExceptionUnwind, Route::Unwind),
    ];
    assert_eq!(cfg.outgoing(entry).collect::<Vec<_>>(), edges);
    assert_eq!(cfg.edge(edges[0]).payload(), &Route::Handler { order: 0 });
    assert_eq!(cfg.edge(edges[1]).payload(), &Route::Handler { order: 1 });
    assert_eq!(cfg.edge(edges[2]).payload(), &Route::Unwind);
}

struct FallibleReach {
    entry: NodeId,
    reject: Option<NodeId>,
}

impl TryEdgeProblem<KeyedGraph<u32, u32, Route>> for FallibleReach {
    type Fact = Option<u8>;
    type Error = &'static str;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &KeyedGraph<u32, u32, Route>) -> Self::Fact {
        None
    }

    fn boundary(
        &self,
        _graph: &KeyedGraph<u32, u32, Route>,
        node: NodeId,
    ) -> Result<Option<Self::Fact>, Self::Error> {
        Ok((node == self.entry).then_some(Some(1)))
    }

    fn meet(
        &self,
        _graph: &KeyedGraph<u32, u32, Route>,
        _node: NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(match (left, right) {
            (Some(left), Some(right)) => Some((*left).max(*right)),
            (Some(value), None) | (None, Some(value)) => Some(*value),
            (None, None) => None,
        })
    }

    fn transfer_node(
        &self,
        _graph: &KeyedGraph<u32, u32, Route>,
        node: NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        if self.reject == Some(node) {
            return Err("node rejected its incoming fact");
        }
        Ok(input.map(|value| value + 1))
    }
}

#[test]
fn keyed_seeded_dataflow_preserves_bottom_and_consumer_errors() {
    let mut graph = KeyedGraph::<u32, u32, Route>::new();
    let entry = graph.intern(&10);
    let reached = graph.intern(&20);
    let island = graph.intern(&1_000);
    let edge = graph.add_edge(entry, reached, Route::Normal);

    let facts = try_solve_edge_problem_from(
        &graph,
        &FallibleReach {
            entry,
            reject: None,
        },
        &[entry],
    )
    .unwrap();
    assert_eq!(facts.fact_in(entry), &Some(1));
    assert_eq!(facts.fact_out(reached), &Some(3));
    assert_eq!(facts.fact_in(island), &None);
    assert_eq!(facts.fact_on(edge), Some(&Some(2)));
    assert_eq!(graph.edge(edge).data(), &Route::Normal);

    let error = try_solve_edge_problem_from(
        &graph,
        &FallibleReach {
            entry,
            reject: Some(reached),
        },
        &[entry],
    )
    .unwrap_err();
    assert_eq!(
        error,
        TrySolveError::Problem("node rejected its incoming fact")
    );
}
