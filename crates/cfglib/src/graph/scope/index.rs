//! The resolutions of every stored reference, their reverse index, and their
//! incremental update.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::graph::epoch::EpochMarks;

use super::query::{
    ScopeGraphQuery, ScopeQuery, ScopeResolution, ScopeResolutionCandidate, ScopeResolutionConfig,
};
use super::storage::{ScopeDatumId, ScopeGraph, ScopeId, ScopeReferenceId};

#[cfg(test)]
mod tests;

/// Resolution results for every stored reference, plus a reverse index.
///
/// An index from [`compute`](Self::compute) also records, for each
/// reference, the scopes its resolution read, so [`update`](Self::update)
/// can re-resolve only the references that an edit can affect. Two indexes
/// are equal when their resolutions and reverse indexes are; the recorded
/// scopes take no part.
#[derive(Debug, Clone)]
pub struct ScopeResolutionIndex {
    resolutions: Vec<ScopeResolution>,
    references_by_datum: Vec<Vec<ScopeReferenceId>>,
    /// The scopes each resolution read, or `None` for an index built without
    /// recording them.
    reads: Option<Reads>,
}

/// The scopes each resolution read, both ways.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Reads {
    /// For each reference, the scopes its resolution read, in the order it
    /// first read them.
    by_reference: Vec<Vec<ScopeId>>,
    /// For each scope, the references whose resolution read it, ascending.
    readers_by_scope: Vec<Vec<ScopeReferenceId>>,
}

impl PartialEq for ScopeResolutionIndex {
    fn eq(&self, other: &Self) -> bool {
        self.resolutions == other.resolutions
            && self.references_by_datum == other.references_by_datum
    }
}

impl Eq for ScopeResolutionIndex {}

/// What changed in a [`ScopeGraph`] since a [`ScopeResolutionIndex`] was
/// computed or last updated.
///
/// Report a scope when its payload, its data (added, or edited through
/// [`ScopeGraph::datum_mut`]) or its outgoing edges (added, removed, or
/// relabeled) changed: the source scope of an edge, not its target. Report a
/// reference when it was edited through [`ScopeGraph::reference_mut`]; a
/// reference the index has not seen yet is resolved without a report. A
/// scope that was only added needs no report, since a scope that no edge
/// reaches is read by no resolution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScopeGraphChanges {
    /// Scopes whose payload, data, outgoing edges or edge labels changed.
    pub scopes: Vec<ScopeId>,
    /// References that were added or edited.
    pub references: Vec<ScopeReferenceId>,
}

/// What one [`ScopeResolutionIndex::update`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScopeIndexUpdate {
    /// The number of references the update resolved again.
    pub resolved_count: usize,
    /// The references whose resolution changed, ascending. A resolution
    /// compares whole, statistics included; a reference the index had not
    /// seen counts as changed.
    pub changed: Vec<ScopeReferenceId>,
}

impl ScopeResolutionIndex {
    /// Resolve every reference in identity order and build the reverse index.
    #[must_use]
    pub fn compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let mut reader = ScopeReads::new(graph.scope_bound());
        let mut by_reference = Vec::with_capacity(graph.reference_count());
        let resolutions = graph
            .reference_ids()
            .map(|reference| {
                let resolution = reader.resolve(graph, reference, query, config);
                by_reference.push(reader.take());
                resolution
            })
            .collect();
        let mut index = Self::from_resolutions(graph, resolutions);
        let mut readers_by_scope = vec![Vec::new(); graph.scope_bound()];
        for (reference, reads) in graph.reference_ids().zip(&by_reference) {
            for scope in reads {
                readers_by_scope[scope.index()].push(reference);
            }
        }
        index.reads = Some(Reads {
            by_reference,
            readers_by_scope,
        });
        index
    }

    /// Builds the reverse index over `resolutions`, one per reference of
    /// `graph` in identity order, without the scopes they read.
    pub(super) fn from_resolutions<S, L, R, D, Q>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        resolutions: Vec<ScopeResolution>,
    ) -> Self {
        let mut references_by_datum = vec![Vec::new(); graph.datum_count()];
        for (reference, resolution) in graph.reference_ids().zip(&resolutions) {
            for datum in sorted_visible_data(resolution) {
                references_by_datum[datum.index()].push(reference);
            }
        }
        Self {
            resolutions,
            references_by_datum,
            reads: None,
        }
    }

    /// Brings the index up to date with `graph` after the edits that
    /// `changes` reports, re-resolving only the references they can affect.
    ///
    /// `graph` must be the graph the index was computed from, edited since
    /// then only as `changes` says, and `query` and `config` must be the ones
    /// it was computed with. The update re-resolves, in identity order, every
    /// reference whose resolution read a changed scope, every reported
    /// reference, and every reference the index has not seen yet, then
    /// patches the reverse index. The result is equal to a fresh
    /// [`compute`](Self::compute) on the edited graph, provided the query
    /// reads nothing of the graph beyond what its resolutions read (see
    /// [`ScopeGraphQuery`]).
    ///
    /// An update costs the resolution of the affected references, a table
    /// over the scopes, and the reader and reference lists that an affected
    /// reference joins or leaves: an edit that changes no reference's read
    /// scopes or visible data patches no list. Recording costs memory in
    /// proportion to the scopes each resolution read, at most the path
    /// prefixes it expanded.
    ///
    /// An index that holds no record of what its resolutions read, such as
    /// one from [`compute_cached`](Self::compute_cached), resolves every
    /// reference on its first update and records them from then on. The two
    /// forms count their work differently, so that first update can report a
    /// reference as changed whose candidates stayed the same.
    ///
    /// # Panics
    ///
    /// Panics when a reported scope or reference is not part of `graph`.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::cmp::Ordering;
    ///
    /// use cfglib::{
    ///     EdgeRef, ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeGraphChanges, ScopeGraphPathLabel,
    ///     ScopeGraphQuery, ScopeId, ScopePath, ScopeQuery, ScopeResolutionConfig,
    ///     ScopeResolutionIndex,
    /// };
    ///
    /// type Names = ScopeGraph<(), (), (), &'static str, &'static str>;
    ///
    /// /// Lexical lookup of a name through parent edges: the nearest
    /// /// declaration shadows the ones further out.
    /// struct Lexical;
    ///
    /// impl ScopeGraphQuery<(), (), (), &'static str, &'static str> for Lexical {
    ///     type State = ();
    ///
    ///     fn initial_state(&self, _: &Names, _: &ScopeQuery<'_, &'static str>) {}
    ///
    ///     fn step(
    ///         &self,
    ///         _: &Names,
    ///         _: &ScopeQuery<'_, &'static str>,
    ///         _: &ScopePath,
    ///         (): &(),
    ///         _: EdgeRef<'_, ScopeId, ScopeEdgeId, ()>,
    ///     ) -> Option<()> {
    ///         Some(())
    ///     }
    ///
    ///     fn accepts(
    ///         &self,
    ///         graph: &Names,
    ///         input: &ScopeQuery<'_, &'static str>,
    ///         _: &ScopePath,
    ///         (): &(),
    ///         datum: ScopeDatumId,
    ///     ) -> bool {
    ///         graph.datum(datum).data() == input.data()
    ///     }
    ///
    ///     fn compare_labels(
    ///         &self,
    ///         _: &Names,
    ///         _: &ScopeQuery<'_, &'static str>,
    ///         left: ScopeGraphPathLabel<'_, ()>,
    ///         right: ScopeGraphPathLabel<'_, ()>,
    ///     ) -> Option<Ordering> {
    ///         let is_edge = |label| matches!(label, ScopeGraphPathLabel::Edge(()));
    ///         Some(is_edge(left).cmp(&is_edge(right)))
    ///     }
    /// }
    ///
    /// let mut graph = Names::new();
    /// let module = graph.add_scope(());
    /// let function = graph.add_scope(());
    /// let other = graph.add_scope(());
    /// graph.add_edge(function, module, ());
    /// graph.add_edge(other, module, ());
    /// let global = graph.add_datum(module, (), "x");
    /// let inside = graph.add_reference(function, "x");
    /// let outside = graph.add_reference(other, "x");
    /// let config = ScopeResolutionConfig::new();
    /// let mut index = ScopeResolutionIndex::compute(&graph, &Lexical, config);
    /// assert_eq!(index.references_to(global), [inside, outside]);
    ///
    /// // A local `x` in the function shadows the global one there. Only the
    /// // reference in the function read the function's scope.
    /// let local = graph.add_datum(function, (), "x");
    /// let changes = ScopeGraphChanges { scopes: vec![function], references: vec![] };
    /// let update = index.update(&graph, &Lexical, config, &changes);
    /// assert_eq!(update.resolved_count, 1);
    /// assert_eq!(update.changed, [inside]);
    /// assert_eq!(index.references_to(local), [inside]);
    /// assert_eq!(index.references_to(global), [outside]);
    /// assert_eq!(index, ScopeResolutionIndex::compute(&graph, &Lexical, config));
    /// ```
    pub fn update<S, L, R, D, Q, P>(
        &mut self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        query: &P,
        config: ScopeResolutionConfig,
        changes: &ScopeGraphChanges,
    ) -> ScopeIndexUpdate
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let known = self.resolutions.len();
        let reference_count = graph.reference_count();
        let reads = self.reads.get_or_insert_with(|| Reads {
            by_reference: Vec::new(),
            readers_by_scope: Vec::new(),
        });
        // Without a record of what the known references read, every one of
        // them may be affected.
        let first = reads.by_reference.len() < known;
        reads.by_reference.resize(reference_count, Vec::new());
        reads
            .readers_by_scope
            .resize(graph.scope_bound(), Vec::new());
        self.references_by_datum
            .resize(graph.datum_count(), Vec::new());

        let affected: Vec<ScopeReferenceId> = if first {
            graph.reference_ids().collect()
        } else {
            let mut affected: Vec<_> = changes
                .scopes
                .iter()
                .flat_map(|scope| reads.readers_by_scope[scope.index()].iter().copied())
                .chain(changes.references.iter().copied())
                .chain((known..reference_count).map(ScopeReferenceId::from_index))
                .collect();
            affected.sort_unstable();
            affected.dedup();
            affected
        };

        // The reader and reference lists that an affected reference leaves
        // or joins; references are handled in identity order, so each patch
        // lists them ascending.
        let mut reader = ScopeReads::new(graph.scope_bound());
        let mut reader_patches: BTreeMap<ScopeId, Patch> = BTreeMap::new();
        let mut datum_patches: BTreeMap<ScopeDatumId, Patch> = BTreeMap::new();
        let mut differing = Vec::new();
        for &reference in &affected {
            let resolution = reader.resolve(graph, reference, query, config);
            let new_reads = reader.take();
            let old_reads = mem::replace(&mut reads.by_reference[reference.index()], new_reads);
            patch_difference(
                &mut reader_patches,
                reference,
                &sorted(old_reads),
                &sorted(reads.by_reference[reference.index()].clone()),
            );
            let old_data = self
                .resolutions
                .get(reference.index())
                .map_or_else(Vec::new, sorted_visible_data);
            patch_difference(
                &mut datum_patches,
                reference,
                &old_data,
                &sorted_visible_data(&resolution),
            );
            match self.resolutions.get_mut(reference.index()) {
                Some(old) if *old == resolution => {}
                Some(old) => {
                    *old = resolution;
                    differing.push(reference);
                }
                None => {
                    self.resolutions.push(resolution);
                    differing.push(reference);
                }
            }
        }
        for (scope, patch) in reader_patches {
            patch.apply(&mut reads.readers_by_scope[scope.index()]);
        }
        for (datum, patch) in datum_patches {
            patch.apply(&mut self.references_by_datum[datum.index()]);
        }
        ScopeIndexUpdate {
            resolved_count: affected.len(),
            changed: differing,
        }
    }

    /// Borrow one reference's resolution.
    ///
    /// # Panics
    ///
    /// Panics when `reference` was not present in the resolved graph.
    #[must_use]
    pub fn resolution(&self, reference: ScopeReferenceId) -> &ScopeResolution {
        &self.resolutions[reference.index()]
    }

    /// References with a visible path to `datum`, in reference identity order.
    ///
    /// Each reference occurs at most once even when it has several visible
    /// paths to the same datum.
    ///
    /// # Panics
    ///
    /// Panics when `datum` was not present in the resolved graph.
    #[must_use]
    pub fn references_to(&self, datum: ScopeDatumId) -> &[ScopeReferenceId] {
        &self.references_by_datum[datum.index()]
    }

    /// Results in reference identity order.
    #[must_use]
    pub fn resolutions(&self) -> &[ScopeResolution] {
        &self.resolutions
    }
}

/// Records the scopes that one resolution reads, each once, in the order it
/// first reads them.
pub(super) struct ScopeReads {
    seen: EpochMarks,
    scopes: Vec<ScopeId>,
}

impl ScopeReads {
    fn new(scope_bound: usize) -> Self {
        Self {
            seen: EpochMarks::new(scope_bound),
            scopes: Vec::new(),
        }
    }

    /// Records that the resolution read `scope`.
    pub(super) fn read(&mut self, scope: ScopeId) {
        if !self.seen.is_marked(scope.index()) {
            self.seen.mark(scope.index());
            self.scopes.push(scope);
        }
    }

    /// Resolves `reference`, recording the scopes its resolution reads.
    fn resolve<S, L, R, D, Q, P>(
        &mut self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        reference: ScopeReferenceId,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> ScopeResolution
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        ScopeResolution::compute_query(
            graph,
            &ScopeQuery::from_reference(graph, reference),
            query,
            config,
            Some(self),
        )
    }

    /// Takes the scopes recorded since the last call.
    fn take(&mut self) -> Vec<ScopeId> {
        self.seen.reset();
        mem::take(&mut self.scopes)
    }
}

/// `items` in ascending order.
fn sorted<T: Ord>(mut items: Vec<T>) -> Vec<T> {
    items.sort_unstable();
    items
}

/// The distinct data a resolution sees, ascending.
fn sorted_visible_data(resolution: &ScopeResolution) -> Vec<ScopeDatumId> {
    let mut data: Vec<_> = resolution
        .candidates()
        .iter()
        .map(ScopeResolutionCandidate::datum)
        .collect();
    data.sort_unstable();
    data.dedup();
    data
}

/// The references that one ascending list loses and gains, each ascending.
#[derive(Default)]
struct Patch {
    removed: Vec<ScopeReferenceId>,
    added: Vec<ScopeReferenceId>,
}

impl Patch {
    /// Applies the patch to `list`, which stays ascending.
    fn apply(self, list: &mut Vec<ScopeReferenceId>) {
        let old = mem::take(list);
        let mut kept = old
            .into_iter()
            .filter(|reference| self.removed.binary_search(reference).is_err())
            .peekable();
        let mut added = self.added.into_iter().peekable();
        loop {
            let next = match (kept.peek(), added.peek()) {
                (Some(&left), Some(&right)) if left < right => kept.next(),
                (_, Some(_)) => added.next(),
                (Some(_), None) => kept.next(),
                (None, None) => break,
            };
            list.extend(next);
        }
    }
}

/// Records in `patches` that `reference` leaves the list of each key in
/// `old` but not in `new`, and joins the list of each key in `new` but not
/// in `old`. Both are ascending.
fn patch_difference<K: Copy + Ord>(
    patches: &mut BTreeMap<K, Patch>,
    reference: ScopeReferenceId,
    old: &[K],
    new: &[K],
) {
    let (mut old, mut new) = (old.iter().peekable(), new.iter().peekable());
    loop {
        match (old.peek(), new.peek()) {
            (Some(left), Some(right)) if left == right => {
                old.next();
                new.next();
            }
            (Some(&&left), Some(&&right)) if left < right => {
                patches.entry(left).or_default().removed.push(reference);
                old.next();
            }
            (Some(&&left), None) => {
                patches.entry(left).or_default().removed.push(reference);
                old.next();
            }
            (_, Some(&&right)) => {
                patches.entry(right).or_default().added.push(reference);
                new.next();
            }
            (None, None) => break,
        }
    }
}
