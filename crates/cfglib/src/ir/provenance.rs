//! Many-to-many provenance between source spans and IR entities.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};

use super::dialect::Vocabulary;

/// A provenance span was empty, reversed, or outside the source model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceError {
    message: &'static str,
}

impl ProvenanceError {
    pub(crate) const fn new(message: &'static str) -> Self {
        Self { message }
    }

    /// Returns the violated provenance invariant.
    #[must_use]
    pub const fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid provenance: {}", self.message)
    }
}

impl core::error::Error for ProvenanceError {}

/// One source span mapped to one IR entity.
#[derive(Debug, Clone)]
pub struct ProvenanceEntry<D: Vocabulary, Entity> {
    /// Source span represented by the IR entity.
    pub source: D::SourceSpan,
    /// Generated IR entity represented by the source span.
    pub entity: Entity,
}

impl<D: Vocabulary, Entity: Copy> Copy for ProvenanceEntry<D, Entity> where D::SourceSpan: Copy {}

impl<D: Vocabulary, Entity: PartialEq> PartialEq for ProvenanceEntry<D, Entity> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.entity == other.entity
    }
}

impl<D: Vocabulary, Entity: Eq> Eq for ProvenanceEntry<D, Entity> {}

impl<D: Vocabulary, Entity: Ord> PartialOrd for ProvenanceEntry<D, Entity> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Vocabulary, Entity: Ord> Ord for ProvenanceEntry<D, Entity> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.source
            .cmp(&other.source)
            .then_with(|| self.entity.cmp(&other.entity))
    }
}

impl<D: Vocabulary, Entity: Hash> Hash for ProvenanceEntry<D, Entity>
where
    D::SourceSpan: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.entity.hash(state);
    }
}

/// Deterministic many-to-many source-to-IR provenance.
///
/// Overlapping spans and multiple entities per span are intentional: one
/// source operation can expand into several IR entities, and one IR entity
/// can represent fused source operations. Synthetic entities have no entry.
///
/// The entity vocabulary is level-specific —
/// [`ir::mlil::EntityId`](crate::ir::mlil::EntityId) for MLIL and
/// [`ir::hlil::EntityId`](crate::ir::hlil::EntityId) for HLIL — while the
/// span model comes from the shared [`Vocabulary`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceMap<D: Vocabulary, Entity> {
    source: D::Source,
    entries: Vec<ProvenanceEntry<D, Entity>>,
    /// Reverse index: the spans recorded for each entity, in the same order
    /// [`entries`](Self::entries) lists them. Answering "what does this
    /// entity come from?" is the common direction, and a scan of every entry
    /// is the wrong shape for it.
    spans_by_entity: BTreeMap<Entity, Vec<D::SourceSpan>>,
}

impl<D: Vocabulary, Entity: Copy + Ord> ProvenanceMap<D, Entity> {
    /// Creates an empty map for one source function.
    #[must_use]
    pub const fn new(source: D::Source) -> Self {
        Self {
            source,
            entries: Vec::new(),
            spans_by_entity: BTreeMap::new(),
        }
    }

    /// Returns the source function identity and coordinate system.
    #[must_use]
    pub const fn source(&self) -> &D::Source {
        &self.source
    }

    /// Adds one mapping in deterministic span-then-entity order.
    ///
    /// Returns `true` for a new entry and `false` for an exact duplicate.
    ///
    /// # Errors
    ///
    /// Returns an error when `source` is empty or reversed.
    pub fn insert(
        &mut self,
        source: D::SourceSpan,
        entity: Entity,
    ) -> Result<bool, ProvenanceError> {
        if D::span_is_empty(&source) {
            return Err(ProvenanceError::new("source span is empty or reversed"));
        }
        let entry = ProvenanceEntry { source, entity };
        let Err(position) = self.entries.binary_search(&entry) else {
            return Ok(false);
        };
        let spans = self.spans_by_entity.entry(entity).or_default();
        if let Err(slot) = spans.binary_search(&entry.source) {
            spans.insert(slot, entry.source.clone());
        }
        self.entries.insert(position, entry);
        Ok(true)
    }

    /// Returns all mappings in deterministic order.
    #[must_use]
    pub fn entries(&self) -> &[ProvenanceEntry<D, Entity>] {
        &self.entries
    }

    /// Returns mappings whose source span contains `point`.
    pub fn mappings_from(
        &self,
        point: D::SourcePoint,
    ) -> impl Iterator<Item = &ProvenanceEntry<D, Entity>> {
        self.entries
            .iter()
            .filter(move |entry| D::span_contains(&entry.source, &point))
    }

    /// Returns mappings that identify `entity`, in span order.
    ///
    /// Answered from the reverse index, so the cost is the entity lookup plus
    /// one binary search per span rather than a scan of every mapping.
    pub fn mappings_to(&self, entity: Entity) -> impl Iterator<Item = &ProvenanceEntry<D, Entity>> {
        self.spans_by_entity
            .get(&entity)
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .filter_map(move |source| {
                let probe = ProvenanceEntry {
                    source: source.clone(),
                    entity,
                };
                let position = self.entries.binary_search(&probe).ok()?;
                self.entries.get(position)
            })
    }

    /// Returns whether no source correspondence has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of distinct mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
