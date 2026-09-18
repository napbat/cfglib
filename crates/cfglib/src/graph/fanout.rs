//! Built-once key tables whose value runs all live in one array.
//!
//! Three shapes over one representation: a key column beside the
//! compressed-sparse-row columns that hold every key's values end to end.
//! [`Fanout`] keys those runs by a sparse ordered key, [`DenseFanout`] by an
//! index into a dense `0..bound` key space, and [`SortedMap`] is the
//! degenerate case in which a key owns one value rather than a run — the same
//! sorted key column, so it is documented, built, and tested beside them
//! rather than in a second module that would repeat the story.
//!
//! # Why this lives beside the store
//!
//! This is [`Graph`](crate::Graph)'s compressed base with the edges taken out
//! of it. A store's outgoing adjacency *is* a dense fan-out — a node index to
//! a run of edge identities in one flat array — and the sparse form is that
//! same table for keys that do not fill a dense space: a symbol to the sites
//! that reference it, a scope to the rows declared in it, a file to its flow
//! sites. Those tables get hand-rolled per consumer, once per index, which is
//! what this module exists to stop; keeping them next to the store that
//! already owns the shape says what they are, where a utility module named for
//! nothing in particular would not.
//!
//! # Built once, then read
//!
//! Every table here is constructed from all of its pairs at once and is
//! read-only afterwards. That is what buys the representation: exact-sized
//! columns, no per-key allocation, a binary search or a direct index instead
//! of a tree walk, and ascending iteration straight down an array. A table
//! that must keep accepting insertions wants a
//! [`BTreeMap`](alloc::collections::BTreeMap); one built from what an
//! analysis just recorded wants this.
//!
//! # Capacity
//!
//! Run bounds are `u32`, so one table holds at most `u32::MAX` values. That is
//! the dense-space contract [`Graph`](crate::Graph) states for its slots, and
//! it is enforced the same way: a panic while building, never a silent
//! truncation.
//!
//! # Why the store's base is still its own
//!
//! [`Graph`](crate::Graph) keeps its four adjacency columns rather than two
//! [`DenseFanout`]s. Its base run is only half of an adjacency answer — a walk
//! continues into the delta chain and a redirected endpoint replaces the run
//! wholesale — so the columns are read by a cursor holding delta and
//! relocation state, not through [`DenseFanout::get`]. Folding them into
//! tables would also change the serialized shape of a public type under the
//! `serde` feature, and the two disagree about the empty table: a fan-out has
//! one canonical empty form, while a store distinguishes a fresh base from a
//! compacted empty one and derives `PartialEq` over that difference. Neither
//! is a behavior-preserving refactor, so the store keeps its own counting pass
//! and this module owns the table that consumers build by hand.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::graph::view::DenseId;

/// Narrow a value count to the `u32` run-bound space.
fn run_bound(count: usize) -> u32 {
    u32::try_from(count).expect("fan-out value count exceeds u32::MAX")
}

/// A sparse ordered key to its run of values, every run in one array.
///
/// The table costs three allocations no matter how many keys it holds, which
/// is the point: a per-key `Vec` pays an allocation, a pointer, and a capacity
/// word for every key, and a reverse index over a workspace has millions of
/// them.
///
/// Every key owns at least one value, so [`contains_key`](Self::contains_key)
/// and a non-empty [`get`](Self::get) mean the same thing. Keys are stored
/// once and never copied, so a key may be a `String` or any other owned
/// identity.
///
/// # Examples
///
/// ```
/// use cfglib::Fanout;
///
/// // Pairs arrive in any order; duplicates of one pair collapse.
/// let sites = Fanout::from_pairs(vec![("write", 4), ("read", 1), ("read", 1), ("read", 7)]);
///
/// assert_eq!(sites.get(&"read"), [1, 7]);
/// assert_eq!(sites.get(&"write"), [4]);
/// assert!(sites.get(&"call").is_empty());
/// assert_eq!(sites.len(), 2);
/// assert_eq!(sites.value_count(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fanout<K, V> {
    /// The distinct keys, ascending.
    keys: Box<[K]>,
    /// Run bounds into `values`: empty when there are no keys, and
    /// `keys.len() + 1` ascending offsets otherwise, so key `index` owns
    /// `offsets[index]..offsets[index + 1]`.
    offsets: Box<[u32]>,
    /// Every key's values, the runs in ascending key order.
    values: Box<[V]>,
}

impl<K, V> Fanout<K, V> {
    /// The number of distinct keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the table holds no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The number of values across every run.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Every key, ascending.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.keys.iter()
    }

    /// Every key with its run, ascending by key.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &[V])> {
        self.keys
            .iter()
            .enumerate()
            .map(|(index, key)| (key, self.run(index)))
    }

    /// The bytes of heap this table owns.
    ///
    /// Exact: the three columns are exact-sized, so this is their contents and
    /// nothing else. Heap reached *through* a key or a value belongs to that
    /// key or value and is not counted here.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.keys.len() * size_of::<K>()
            + self.offsets.len() * size_of::<u32>()
            + self.values.len() * size_of::<V>()
    }

    /// The run owned by the key at `index`.
    fn run(&self, index: usize) -> &[V] {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        &self.values[start..end]
    }
}

impl<K: Ord, V> Fanout<K, V> {
    /// Builds the table from grouped values, keeping each group's order.
    ///
    /// Groups may arrive in any key order and a key may be grouped more than
    /// once; the sort is stable and repeated keys concatenate, so a key's run
    /// is every value given for it in the order it was given. A group with no
    /// values contributes no key.
    ///
    /// # Panics
    ///
    /// Panics when the groups hold more than `u32::MAX` values in total.
    #[must_use]
    pub fn from_grouped<G>(groups: impl IntoIterator<Item = (K, G)>) -> Self
    where
        G: IntoIterator<Item = V>,
    {
        // Each group is materialized so the sort moves whole runs rather than
        // individual values; the group vectors are consumed as the columns are
        // filled and none of them outlives the build.
        let mut grouped: Vec<(K, Vec<V>)> = groups
            .into_iter()
            .map(|(key, values)| (key, values.into_iter().collect()))
            .collect();
        grouped.sort_by(|left, right| left.0.cmp(&right.0));

        let value_count = grouped.iter().map(|(_, values)| values.len()).sum();
        let mut columns = Columns::with_capacity(grouped.len(), value_count);
        for (key, values) in grouped {
            if values.is_empty() {
                continue;
            }
            columns.open(key);
            columns.values.extend(values);
        }
        columns.finish()
    }

    /// The values recorded for `key`, or an empty slice when it is absent.
    #[must_use]
    pub fn get(&self, key: &K) -> &[V] {
        self.keys
            .binary_search(key)
            .map_or(&[][..], |index| self.run(index))
    }

    /// Whether the table holds a run for `key`.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.keys.binary_search(key).is_ok()
    }
}

impl<K: Ord, V: Ord> Fanout<K, V> {
    /// Builds the table from pairs in any order.
    ///
    /// Each run ends up ascending and deduplicated: repeated `(key, value)`
    /// pairs collapse to one value, which is what a table built from what an
    /// analysis observed usually wants. Use
    /// [`from_grouped`](Self::from_grouped) to keep a run's own order and its
    /// repetitions.
    ///
    /// # Panics
    ///
    /// Panics when `pairs` holds more than `u32::MAX` distinct pairs.
    #[must_use]
    pub fn from_pairs(mut pairs: Vec<(K, V)>) -> Self {
        pairs.sort_unstable();
        pairs.dedup();

        let mut columns = Columns::with_capacity(0, pairs.len());
        for (key, value) in pairs {
            columns.open(key);
            columns.values.push(value);
        }
        columns.finish()
    }
}

impl<K, V> Default for Fanout<K, V> {
    /// The empty table, which allocates nothing.
    fn default() -> Self {
        Self {
            keys: Box::default(),
            offsets: Box::default(),
            values: Box::default(),
        }
    }
}

/// [`Fanout`]'s columns under construction, keys opened in ascending order.
struct Columns<K, V> {
    keys: Vec<K>,
    offsets: Vec<u32>,
    values: Vec<V>,
}

impl<K: PartialEq, V> Columns<K, V> {
    fn with_capacity(keys: usize, values: usize) -> Self {
        Self {
            keys: Vec::with_capacity(keys),
            offsets: Vec::with_capacity(keys + 1),
            values: Vec::with_capacity(values),
        }
    }

    /// Starts `key`'s run at the current end of the value column, or leaves
    /// the open run alone when the same key repeats.
    fn open(&mut self, key: K) {
        if self.keys.last() == Some(&key) {
            return;
        }
        self.offsets.push(run_bound(self.values.len()));
        self.keys.push(key);
    }

    /// Closes the last run and freezes the columns to their exact sizes.
    fn finish(mut self) -> Fanout<K, V> {
        if !self.keys.is_empty() {
            self.offsets.push(run_bound(self.values.len()));
        }
        Fanout {
            keys: self.keys.into_boxed_slice(),
            offsets: self.offsets.into_boxed_slice(),
            values: self.values.into_boxed_slice(),
        }
    }
}

/// A dense `0..bound` key space to runs of values: the compressed-sparse-row
/// form of a [`Fanout`].
///
/// Lookup is an array index rather than a search, which is what a key space
/// that is already dense — node identities, file indices, scope indices —
/// should cost. Keys with no values are part of the table: they own an empty
/// run, and [`iter`](Self::iter) yields them, exactly as
/// [`Graph::node_ids`](crate::Graph::node_ids) yields a node with no edges.
///
/// `K` is any [`DenseId`], so a tagged store identity keys the table directly
/// and the index conversion stays where the identity is defined:
/// `DenseFanout<RefLoc, Id<SymbolTag>>` is a symbol's references, and the
/// default `usize` covers a plain dense index.
///
/// # Examples
///
/// ```
/// use cfglib::DenseFanout;
///
/// // Per-key order is arrival order: key 2 recorded "c" before "b".
/// let rows = DenseFanout::from_pairs(3, [(2usize, "c"), (0, "a"), (2, "b")]);
///
/// assert_eq!(rows.get(0), ["a"]);
/// assert!(rows.get(1).is_empty());
/// assert_eq!(rows.get(2), ["c", "b"]);
/// assert_eq!(rows.bound(), 3);
/// assert_eq!(rows.value_count(), 3);
/// ```
///
/// A tagged identity keys the same table:
///
/// ```
/// use cfglib::{DenseFanout, Graph, NodeId};
///
/// let mut graph = Graph::<&str, ()>::new();
/// let definition = graph.add_node("definition");
/// let call = graph.add_node("call");
///
/// let comments: DenseFanout<&str, NodeId> =
///     DenseFanout::from_pairs(graph.node_bound(), [(call, "tail call"), (call, "hot")]);
/// assert!(comments.get(definition).is_empty());
/// assert_eq!(comments.get(call), ["tail call", "hot"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseFanout<V, K: DenseId = usize> {
    /// Run bounds into `values`: empty when the key space is empty, and
    /// `bound + 1` ascending offsets otherwise, so key `index` owns
    /// `offsets[index]..offsets[index + 1]`.
    offsets: Box<[u32]>,
    /// Every key's values, the runs in ascending key order.
    values: Box<[V]>,
    key: PhantomData<fn() -> K>,
}

impl<V, K: DenseId> DenseFanout<V, K> {
    /// Builds the table over the `0..bound` key space from pairs in any order.
    ///
    /// One counting pass and one placing pass, so building costs
    /// `O(pairs + bound)` rather than a sort, and each key's run keeps the
    /// order its pairs arrived in.
    ///
    /// # Panics
    ///
    /// Panics when a key is not below `bound`, or when `pairs` holds more than
    /// `u32::MAX` values.
    #[must_use]
    pub fn from_pairs(bound: usize, pairs: impl IntoIterator<Item = (K, V)>) -> Self {
        let pairs: Vec<(K, V)> = pairs.into_iter().collect();
        // Every run bound is a `u32`, so the total is checked once here,
        // before the counting pass writes it into the offsets column.
        let value_count = run_bound(pairs.len()) as usize;
        if bound == 0 {
            assert!(
                pairs.is_empty(),
                "dense fan-out key is outside the empty key space"
            );
            return Self::default();
        }

        let mut offsets = vec![0_u32; bound + 1];
        for (key, _) in &pairs {
            let index = key.index();
            assert!(index < bound, "dense fan-out key is outside 0..{bound}");
            offsets[index + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }

        // Placing moves each value to the slot its key's cursor names, and a
        // slot can only be written through an owned `Option`; every slot is
        // written exactly once, because the counting pass reserved one per
        // pair.
        let mut cursors: Vec<u32> = offsets[..bound].to_vec();
        let mut slots: Vec<Option<V>> = Vec::with_capacity(value_count);
        slots.resize_with(value_count, || None);
        for (key, value) in pairs {
            let cursor = &mut cursors[key.index()];
            slots[*cursor as usize] = Some(value);
            *cursor += 1;
        }
        let values = slots
            .into_iter()
            .map(|slot| slot.expect("every counted pair filled its reserved slot"))
            .collect::<Vec<_>>();

        Self {
            offsets: offsets.into_boxed_slice(),
            values: values.into_boxed_slice(),
            key: PhantomData,
        }
    }

    /// The exclusive upper bound of the key space.
    #[must_use]
    pub fn bound(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The number of values across every run.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// The values recorded for `key`, in the order they arrived.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not below [`bound`](Self::bound). A key of the
    /// table's own space that recorded nothing is not an error: it owns an
    /// empty run.
    #[must_use]
    pub fn get(&self, key: K) -> &[V] {
        let index = key.index();
        assert!(
            index < self.bound(),
            "dense fan-out key is outside 0..{}",
            self.bound()
        );
        self.run(index)
    }

    /// Every key of the space with its run, ascending, empty runs included.
    pub fn iter(&self) -> impl Iterator<Item = (K, &[V])> {
        (0..self.bound()).map(|index| (K::from_index(index), self.run(index)))
    }

    /// The bytes of heap this table owns.
    ///
    /// Exact: both columns are exact-sized, so this is their contents and
    /// nothing else. Heap reached *through* a value belongs to that value and
    /// is not counted here.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.offsets.len() * size_of::<u32>() + self.values.len() * size_of::<V>()
    }

    /// The run owned by the key at `index`.
    fn run(&self, index: usize) -> &[V] {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        &self.values[start..end]
    }
}

impl<V, K: DenseId> Default for DenseFanout<V, K> {
    /// The table over the empty key space, which allocates nothing.
    fn default() -> Self {
        Self {
            offsets: Box::default(),
            values: Box::default(),
            key: PhantomData,
        }
    }
}

/// A sparse ordered key to one value, in one sorted array.
///
/// The single-value case of a [`Fanout`], and the built-once counterpart of a
/// [`BTreeMap`](alloc::collections::BTreeMap): the same lookups and the same
/// ascending iteration out of one allocation instead of a tree of nodes.
/// Because it is built from pairs rather than inserted into, the caller says
/// what a repeated key keeps.
///
/// # Examples
///
/// ```
/// use cfglib::SortedMap;
///
/// // `|slot, value| *slot = value` is what repeated `insert` calls mean.
/// let last = SortedMap::from_pairs(vec![("a", 1), ("b", 2), ("a", 3)], |slot, value| {
///     *slot = value;
/// });
/// assert_eq!(last.get(&"a"), Some(&3));
///
/// // Any other fold over a key's pairs is equally available.
/// let total = SortedMap::from_pairs(vec![("a", 1), ("b", 2), ("a", 3)], |slot, value| {
///     *slot += value;
/// });
/// assert_eq!(total.get(&"a"), Some(&4));
/// assert_eq!(total.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedMap<K, V> {
    /// One entry per distinct key, ascending.
    entries: Box<[(K, V)]>,
}

impl<K, V> SortedMap<K, V> {
    /// The number of distinct keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every key with its value, ascending by key.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }

    /// The bytes of heap this map owns.
    ///
    /// Exact: the entry column is exact-sized, so this is its contents and
    /// nothing else. Heap reached *through* a key or a value belongs to that
    /// key or value and is not counted here.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.entries.len() * size_of::<(K, V)>()
    }
}

impl<K: Ord, V> SortedMap<K, V> {
    /// Builds the map from pairs in any order, folding repeated keys with
    /// `merge`.
    ///
    /// The sort is stable, so `merge` sees a key's pairs in the order they
    /// were recorded and folds them left to right: `|slot, value| *slot =
    /// value` reproduces repeated `BTreeMap::insert` calls, and a field-wise
    /// merge keeps several recorded facts about one key.
    #[must_use]
    pub fn from_pairs(mut pairs: Vec<(K, V)>, mut merge: impl FnMut(&mut V, V)) -> Self {
        pairs.sort_by(|left, right| left.0.cmp(&right.0));

        let mut entries: Vec<(K, V)> = Vec::with_capacity(pairs.len());
        for (key, value) in pairs {
            match entries.last_mut() {
                Some(last) if last.0 == key => merge(&mut last.1, value),
                _ => entries.push((key, value)),
            }
        }
        Self {
            entries: entries.into_boxed_slice(),
        }
    }

    /// The value recorded for `key`, or `None` when it is absent.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self
            .entries
            .binary_search_by(|entry| entry.0.cmp(key))
            .ok()?;
        Some(&self.entries[index].1)
    }

    /// Whether the map holds an entry for `key`.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries
            .binary_search_by(|entry| entry.0.cmp(key))
            .is_ok()
    }
}

impl<K, V> Default for SortedMap<K, V> {
    /// The empty map, which allocates nothing.
    fn default() -> Self {
        Self {
            entries: Box::default(),
        }
    }
}

#[cfg(test)]
mod tests;
