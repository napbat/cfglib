//! Oracles for the shape of a tree and the honesty of its report: nodes form a
//! tree, a visitor sees them in a known order, and a report names exactly the
//! gotos the tree holds.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, Cfg, ControlDescent, ControlNode, ControlNodeId, ControlTree, ControlVisitor,
    LiftReport,
};

use crate::coverage::blocks_under;

/// The shape invariants every tree has: preorder visits each node once and
/// reaches all of them, a node has exactly one parent, and the root is the last
/// node allocated.
pub fn assert_well_formed(tree: &ControlTree) {
    let order: Vec<ControlNodeId> = tree.preorder().collect();
    let unique: BTreeSet<ControlNodeId> = order.iter().copied().collect();
    assert_eq!(unique.len(), order.len(), "preorder repeated a node");
    assert_eq!(
        order.len(),
        tree.node_count(),
        "every node is reachable from the root"
    );
    assert_eq!(order.first().copied(), Some(tree.root()));
    assert_eq!(tree.node_ids().len(), tree.node_count());
    assert_eq!(tree.root().index(), tree.node_count() - 1);

    let mut parent: BTreeMap<ControlNodeId, ControlNodeId> = BTreeMap::new();
    for id in tree.node_ids() {
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                assert_eq!(parent.insert(child, id), None, "{child:?} has two parents");
            }
        }
    }
    for id in tree.node_ids() {
        assert_eq!(
            parent.contains_key(&id),
            id != tree.root(),
            "only the root is parentless"
        );
    }

    // Preorder is the parent-before-child order: a parent precedes the nodes
    // it holds.
    let position: BTreeMap<ControlNodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect();
    for (&child, &owner) in &parent {
        assert!(position[&owner] < position[&child]);
    }
}

/// Exception edges are not drawn in a tree: they are what nesting means. Every
/// handler or unwind edge runs from a block under some `TryCatch`'s protected
/// body to the entry of one of that node's handlers (or into its finally body).
pub fn assert_exception_edges_are_expressed_by_nesting<I, E>(cfg: &Cfg<I, E>, tree: &ControlTree) {
    let mut regions: Vec<(BTreeSet<BlockId>, BTreeSet<BlockId>)> = Vec::new();
    for id in tree.preorder() {
        if let ControlNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        } = *tree.node(id)
        {
            let mut entered: BTreeSet<BlockId> = tree
                .handlers(handlers)
                .iter()
                .map(|handler| handler.entry)
                .collect();
            entered.extend(blocks_under(tree, finally_body));
            regions.push((blocks_under(tree, try_body), entered));
        }
    }
    for edge in cfg.edges() {
        if !matches!(
            edge.kind(),
            cfglib::EdgeKind::ExceptionHandler | cfglib::EdgeKind::ExceptionUnwind
        ) {
            continue;
        }
        assert!(
            regions.iter().any(|(protected, entered)| {
                protected.contains(&edge.source()) && entered.contains(&edge.target())
            }),
            "{} -> {} is not expressed by any try/catch nesting",
            edge.source(),
            edge.target()
        );
    }
}

/// The documented allocation order: every child has a lower handle than the
/// node that holds it, so a bottom-up fold over `node_ids` sees children first.
pub fn assert_children_precede_parents(tree: &ControlTree) {
    for id in tree.node_ids() {
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                assert!(
                    child < id,
                    "{child:?} is allocated after the {:?} that holds it",
                    tree.node(id)
                );
            }
        }
    }
}

/// The nodes a tree holds of one kind.
pub fn nodes_where(
    tree: &ControlTree,
    mut keep: impl FnMut(&ControlNode) -> bool,
) -> Vec<ControlNodeId> {
    tree.preorder().filter(|&id| keep(tree.node(id))).collect()
}

/// The targets of every `Goto` node, in preorder.
pub fn goto_targets_in_tree(tree: &ControlTree) -> Vec<BlockId> {
    tree.preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Goto { target } => Some(target),
            _ => None,
        })
        .collect()
}

/// The blocks a `Label` node names, in preorder.
pub fn label_blocks_in_tree(tree: &ControlTree) -> Vec<BlockId> {
    tree.preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Label { block, .. } => Some(block),
            _ => None,
        })
        .collect()
}

/// A report is the tree's own account of its gotos: one diagnostic per `Goto`
/// node, naming the same target, and a label for each target unless the report
/// lists it as unresolved.
pub fn assert_report_matches_tree(tree: &ControlTree, report: &LiftReport) {
    let mut from_tree = goto_targets_in_tree(tree);
    let mut from_report: Vec<BlockId> = report.gotos.iter().map(|goto| goto.target).collect();
    from_tree.sort();
    from_report.sort();
    assert_eq!(from_tree, from_report);

    let labeled: BTreeSet<BlockId> = label_blocks_in_tree(tree).into_iter().collect();
    for target in from_tree {
        assert!(
            labeled.contains(&target) || report.unresolved_labels.contains(&target),
            "goto to {target} has no label and is not reported unresolved"
        );
    }
}

/// A fully structured report is exactly the one with nothing to say, and the
/// tree holds no `Goto` to back that up.
pub fn assert_fully_structured(tree: &ControlTree, report: &LiftReport) {
    assert!(report.is_fully_structured(), "{report:?}");
    assert_eq!(report.gotos, []);
    assert_eq!(report.unresolved_labels, []);
    assert_eq!(report.unstructured_regions, []);
    assert_eq!(goto_targets_in_tree(tree), []);
}

/// A visitor that records the order of `enter` and `leave` calls.
#[derive(Default)]
pub struct EventLog {
    pub events: Vec<(bool, ControlNodeId)>,
    pub skip: BTreeSet<ControlNodeId>,
}

impl ControlVisitor for EventLog {
    fn enter(&mut self, _tree: &ControlTree, node: ControlNodeId) -> ControlDescent {
        self.events.push((true, node));
        if self.skip.contains(&node) {
            ControlDescent::Skip
        } else {
            ControlDescent::Into
        }
    }

    fn leave(&mut self, _tree: &ControlTree, node: ControlNodeId) {
        self.events.push((false, node));
    }
}

/// The nodes of a tree in preorder, by plain recursion over `child_bodies`.
pub fn reference_preorder(tree: &ControlTree) -> Vec<ControlNodeId> {
    fn visit(tree: &ControlTree, id: ControlNodeId, out: &mut Vec<ControlNodeId>) {
        out.push(id);
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                visit(tree, child, out);
            }
        }
    }
    let mut out = Vec::new();
    visit(tree, tree.root(), &mut out);
    out
}

/// The nodes of a tree in postorder, by plain recursion.
pub fn reference_postorder(tree: &ControlTree) -> Vec<ControlNodeId> {
    fn visit(tree: &ControlTree, id: ControlNodeId, out: &mut Vec<ControlNodeId>) {
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                visit(tree, child, out);
            }
        }
        out.push(id);
    }
    let mut out = Vec::new();
    visit(tree, tree.root(), &mut out);
    out
}

/// How deep a node sits, counting the root as 1, by plain recursion.
pub fn reference_depth(tree: &ControlTree, id: ControlNodeId) -> usize {
    1 + tree
        .child_bodies(id)
        .flat_map(|body| tree.body(body).to_vec())
        .map(|child| reference_depth(tree, child))
        .max()
        .unwrap_or(0)
}
