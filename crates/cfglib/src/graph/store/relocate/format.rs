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
use alloc::vec::Vec;

use serde::de::Deserializer;
use serde::ser::{SerializeMap, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use super::super::adjacency::{MAX_SLOTS, NONE};
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
#[derive(Deserialize)]
#[serde(rename = "Relocations")]
struct Encoded {
    outgoing: BTreeMap<u32, Vec<u32>>,
    incoming: BTreeMap<u32, Vec<u32>>,
}

impl Runs {
    /// The index for `encoded`, sized to its highest node.
    ///
    /// The encoding does not say how many nodes the store has, and the index
    /// is allocated from its keys, so a key that cannot name a node, or one
    /// whose index cannot be allocated, is an error rather than an abort.
    fn from_encoded<E: serde::de::Error>(encoded: BTreeMap<u32, Vec<u32>>) -> Result<Self, E> {
        let Some((&highest, _)) = encoded.last_key_value() else {
            return Ok(Self::default());
        };
        if highest as usize >= MAX_SLOTS {
            return Err(E::custom(format_args!(
                "relocated node {highest} is outside the dense identity space"
            )));
        }
        let bound = highest as usize + 1;
        let mut index = Vec::new();
        index
            .try_reserve_exact(bound)
            .map_err(|_| E::custom("relocation index does not fit in memory"))?;
        index.resize(bound, NONE);
        let mut runs = Vec::with_capacity(encoded.len());
        for (node, run) in encoded {
            index[node as usize] = raw_index(runs.len());
            runs.push(run);
        }
        Ok(Self { index, runs })
    }
}

impl<'de> Deserialize<'de> for Relocations {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Encoded { outgoing, incoming } = Encoded::deserialize(deserializer)?;
        Ok(Self {
            outgoing: Runs::from_encoded(outgoing)?,
            incoming: Runs::from_encoded(incoming)?,
        })
    }
}
