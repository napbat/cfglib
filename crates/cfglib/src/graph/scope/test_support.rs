//! Fixtures shared by the scope-graph tests: one label alphabet with its
//! specificity order, a chain of parent scopes, and random graphs.

extern crate alloc;

use alloc::vec::Vec;

use super::{ScopeGraph, ScopeGraphPathLabel, ScopeId};
use crate::test_util::rng::Rng;

/// The edge labels of the test graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Label {
    /// A lexical parent.
    Parent,
    /// An import.
    Import,
    /// An edge that no lookup takes.
    Hidden,
    /// The letters of a path language that comes back to a scope.
    A,
    B,
}

/// The specificity of a label: a declaration shadows an import (or a
/// letter), which shadows a parent, which shadows a hidden edge.
pub(super) const fn rank(label: ScopeGraphPathLabel<'_, Label>) -> u8 {
    match label {
        ScopeGraphPathLabel::Declaration => 0,
        ScopeGraphPathLabel::Edge(Label::Import | Label::A | Label::B) => 1,
        ScopeGraphPathLabel::Edge(Label::Parent) => 2,
        ScopeGraphPathLabel::Edge(Label::Hidden) => 3,
    }
}

/// A chain of `count` scopes with the default payload, each with a parent
/// edge to the one before it, so every parent edge points upward.
pub(super) fn chain<S: Default, R, D, Q>(
    count: usize,
) -> (ScopeGraph<S, Label, R, D, Q>, Vec<ScopeId>) {
    let mut graph = ScopeGraph::new();
    let scopes: Vec<ScopeId> = (0..count).map(|_| graph.add_scope(S::default())).collect();
    for pair in scopes.windows(2) {
        graph.add_edge(pair[1], pair[0], Label::Parent);
    }
    (graph, scopes)
}

/// A random name below three.
pub(super) fn name(random: &mut Rng) -> u8 {
    u8::try_from(random.below(3)).expect("a name below three fits in u8")
}

/// A random label: a parent three times in five, otherwise an import.
pub(super) fn parent_or_import(random: &mut Rng) -> Label {
    if random.chance(60) {
        Label::Parent
    } else {
        Label::Import
    }
}

/// A random graph of `1..=max_scopes` scopes with payloads drawn by
/// `payload`. Each scope has up to two edges labeled by
/// [`parent_or_import`], and up to two data and up to two references named
/// by [`name`]. An edge of a scope after the first goes to an earlier scope
/// with a chance of `upward_percent` percent, and to any scope otherwise, so
/// it may close a cycle.
pub(super) fn random_graph<S>(
    random: &mut Rng,
    max_scopes: usize,
    upward_percent: usize,
    mut payload: impl FnMut(&mut Rng) -> S,
) -> ScopeGraph<S, Label, (), u8, u8> {
    let count = 1 + random.below(max_scopes);
    let mut graph = ScopeGraph::new();
    let scopes: Vec<ScopeId> = (0..count)
        .map(|_| graph.add_scope(payload(random)))
        .collect();
    for (index, &scope) in scopes.iter().enumerate() {
        for _ in 0..random.below(3) {
            let target = if index > 0 && random.chance(upward_percent) {
                random.below(index)
            } else {
                random.below(count)
            };
            graph.add_edge(scope, scopes[target], parent_or_import(random));
        }
        for _ in 0..random.below(3) {
            graph.add_datum(scope, (), name(random));
        }
        for _ in 0..random.below(3) {
            graph.add_reference(scope, name(random));
        }
    }
    graph
}
