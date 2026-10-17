//! Reading a stack graph from its serialized form without trusting it.
//!
//! Path extraction and search index the file table by each node's file, walk
//! the per-file node lists, start from the two singletons, and follow the
//! scope a scoped push carries, so a document whose tables disagree would
//! panic or answer for the wrong node in the middle of a query. A document is
//! therefore decoded into [`Stored`], the same columns with no behavior, and
//! becomes a [`StackGraph`] only once its tables have been checked against the
//! nodes and edges of the store underneath, which has already been checked
//! itself.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{
    StackGraph, StackNodeId, StackNodeKind, StackStore, require_attachable_scope, require_joinable,
};

/// The columns of a [`StackGraph`], exactly as they are written.
///
/// A field here without its counterpart in `StackGraph` (or the reverse) does
/// not compile, because [`TryFrom`] below names every one.
#[derive(serde::Deserialize)]
#[serde(bound(
    deserialize = "F: serde::Deserialize<'de>, S: serde::Deserialize<'de>, \
                   N: serde::Deserialize<'de>, E: serde::Deserialize<'de>"
))]
pub(super) struct Stored<F, S, N, E> {
    graph: StackStore<S, N, E>,
    files: Vec<F>,
    file_nodes: Vec<Vec<StackNodeId>>,
    root: StackNodeId,
    jump_to_scope: StackNodeId,
}

impl<F, S, N, E> TryFrom<Stored<F, S, N, E>> for StackGraph<F, S, N, E> {
    type Error = &'static str;

    /// Accepts a graph the builders could have produced, retired nodes and
    /// edges included: the rules are the ones `check` and its parts spell out.
    fn try_from(stored: Stored<F, S, N, E>) -> Result<Self, Self::Error> {
        let Stored {
            graph,
            files,
            file_nodes,
            root,
            jump_to_scope,
        } = stored;
        let graph = Self {
            graph,
            files,
            file_nodes,
            root,
            jump_to_scope,
        };
        graph.check()?;
        Ok(graph)
    }
}

impl<F, S, N, E> StackGraph<F, S, N, E> {
    /// Check the rules `new`, `add_file`, the `add_*_node` builders, `add_edge`,
    /// and `clear_file` keep, in an order that lets each check index only
    /// what the ones before it have bounded.
    fn check(&self) -> Result<(), &'static str> {
        self.check_singletons()?;
        self.check_nodes()?;
        self.check_file_lists()?;
        self.check_edges()
    }

    /// `new` makes the root and the jump-to-scope node, and nothing removes
    /// them.
    fn check_singletons(&self) -> Result<(), &'static str> {
        if !self.contains_node(self.root) || !self.node(self.root).kind().is_root() {
            return Err("the root is not a live root node");
        }
        if !self.contains_node(self.jump_to_scope)
            || !self.node(self.jump_to_scope).kind().is_jump_to_scope()
        {
            return Err("the jump-to-scope node is not a live jump-to-scope node");
        }
        Ok(())
    }

    /// Every slot, live or retired, since `clear_file` keeps a retired node's
    /// file, kind, and payload readable: only the singletons are in no file
    /// and carry no payload, and no other node is a singleton.
    fn check_nodes(&self) -> Result<(), &'static str> {
        for index in 0..self.node_bound() {
            let id = StackNodeId::from_index(index);
            let node = self.node(id);
            if id == self.root || id == self.jump_to_scope {
                if node.file().is_some() || node.payload().is_some() {
                    return Err("a singleton node is in a file or carries a payload");
                }
                continue;
            }
            let Some(file) = node.file().filter(|file| file.index() < self.files.len()) else {
                return Err("a node is in no file, or in one the graph does not hold");
            };
            if node.payload().is_none() {
                return Err("a node of a file carries no payload");
            }
            if node.kind().is_root() || node.kind().is_jump_to_scope() {
                return Err("a second singleton node");
            }
            if let StackNodeKind::PushScopedSymbol { scope, .. } = node.kind()
                && (scope.index() >= self.node_bound()
                    || require_attachable_scope(file, *scope, self.node(*scope)).is_err())
            {
                return Err(
                    "a scoped push names something other than an exported scope of its file",
                );
            }
        }
        Ok(())
    }

    /// `add_file` starts an empty list, `add_local_node` appends each live
    /// node to its file's, and `clear_file` empties it, so a list holds the
    /// live nodes of its file in identity order. Every node's file is below
    /// the file count, which `check_nodes` has established.
    fn check_file_lists(&self) -> Result<(), &'static str> {
        let mut expected = vec![Vec::new(); self.files.len()];
        for node in self.node_ids() {
            if let Some(file) = self.node(node).file() {
                expected[file.index()].push(node);
            }
        }
        if expected != self.file_nodes {
            return Err("the per-file node lists are not the live nodes grouped by file");
        }
        Ok(())
    }

    /// The rule `add_edge` enforces, for every live edge.
    fn check_edges(&self) -> Result<(), &'static str> {
        for edge in self.edge_ids() {
            let edge = self.edge(edge);
            let (source, target) = (edge.source(), edge.target());
            if require_joinable(source, self.node(source), target, self.node(target)).is_err() {
                return Err("an edge leaves the jump-to-scope node or joins two files directly");
            }
        }
        Ok(())
    }
}
