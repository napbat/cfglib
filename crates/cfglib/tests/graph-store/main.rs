//! Consumer scenario: the symbol-index store that symtree-xref, symtree-workspace and semctx keep.
//!
//! They add and remove nodes and edges of a `Graph`, redirect edges without
//! losing their identity, compact when they decide to pay for it, and carry
//! their own identities across the compaction through the `Renumbering` it
//! returns. The same renumbering carries a CFG pass's `Rewrite` forward with
//! `Rewrite::then_renumbered`.
//!
//! Node and edge payloads are deliberately not `Copy`: a consumer's payloads
//! own strings and vectors, and every store operation has to leave them
//! intact.

mod cfg_rewrite;
mod compaction;
mod lifecycle;
mod reference_model;
mod tombstone_differential;

/// A node payload shaped like a symbol-index entry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Symbol {
    name: String,
    file: u32,
}

/// An edge payload shaped like a relation record.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Relation {
    label: String,
    sites: Vec<u32>,
}

fn symbol(name: &str) -> Symbol {
    Symbol {
        name: name.to_owned(),
        file: 7,
    }
}

fn relation(label: &str) -> Relation {
    Relation {
        label: label.to_owned(),
        sites: vec![1, 2, 3],
    }
}
