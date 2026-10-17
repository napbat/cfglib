//! The serialized form of the relocation overlay.
//!
//! The encoding is the one the overlay had when each direction was an ordered
//! map from node to run: a `Relocations` struct of two maps, `outgoing` then
//! `incoming`, each ascending by node and holding every run verbatim. Stores
//! written then still load, and a store written now reads the same to any
//! earlier release. The dense index is rebuilt on load and is never written,
//! so what goes out does not depend on where the index happens to put a node.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use serde::ser::{SerializeMap, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use super::super::adjacency::NONE;
use super::super::raw_index;
use super::{Relocations, Runs};

/// One direction, written as the map it used to be.
struct RunMap<'runs>(&'runs Runs);

impl Serialize for RunMap<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // The length is exact, which formats that prefix a map with its
        // length require, and which a filtered walk of the index cannot tell
        // them.
        let mut map = serializer.serialize_map(Some(self.0.runs.len()))?;
        for (node, run) in self.0.iter() {
            map.serialize_entry(&node, run)?;
        }
        map.end()
    }
}

impl Serialize for Relocations {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Relocations", 2)?;
        state.serialize_field("outgoing", &RunMap(&self.outgoing))?;
        state.serialize_field("incoming", &RunMap(&self.incoming))?;
        state.end()
    }
}

/// What the encoding contains, before it becomes an index.
///
/// It is not an overlay until it has been checked against the store it
/// belongs to, which the encoding does not describe: see
/// [`into_relocations`](Self::into_relocations).
#[derive(Deserialize)]
#[serde(rename = "Relocations")]
pub(in crate::graph::store) struct Encoded {
    outgoing: BTreeMap<u32, Vec<u32>>,
    incoming: BTreeMap<u32, Vec<u32>>,
}

impl Encoded {
    /// The overlay this encodes, for a store of `node_bound` node slots and
    /// `edge_bound` edge slots.
    ///
    /// The index is allocated from the keys, so a run that belongs to no node
    /// of the store, or names an edge it does not hold, is an error rather
    /// than an allocation of the size of whatever the document claims.
    pub(in crate::graph::store) fn into_relocations(
        self,
        node_bound: usize,
        edge_bound: usize,
    ) -> Result<Relocations, &'static str> {
        Ok(Relocations {
            outgoing: Runs::from_encoded(self.outgoing, node_bound, edge_bound)?,
            incoming: Runs::from_encoded(self.incoming, node_bound, edge_bound)?,
        })
    }
}

impl Runs {
    fn from_encoded(
        encoded: BTreeMap<u32, Vec<u32>>,
        node_bound: usize,
        edge_bound: usize,
    ) -> Result<Self, &'static str> {
        let Some((&highest, _)) = encoded.last_key_value() else {
            return Ok(Self::default());
        };
        if highest as usize >= node_bound {
            return Err("an overlay run belongs to a node the store does not hold");
        }
        if encoded
            .values()
            .flatten()
            .any(|&edge| edge as usize >= edge_bound)
        {
            return Err("an overlay run names an edge the store does not hold");
        }
        let mut index = vec![NONE; highest as usize + 1];
        let mut runs = Vec::with_capacity(encoded.len());
        for (node, run) in encoded {
            index[node as usize] = raw_index(runs.len());
            runs.push(run);
        }
        Ok(Self { index, runs })
    }
}
