//! The dense table of the roots that one function names.

extern crate alloc;

use alloc::vec::Vec;
use core::slice;

use crate::ir::rtl::{Dialect, FunctionView, Place, RecordView, Selection, Statement, WriteTarget};

/// The roots that one function names, in ascending order.
///
/// A function names a root when a read selects it, when a write targets it,
/// when it is a candidate of an indexed selection or write, when it is the
/// bank of a sparse access, or when a raise binds it. The position of a root
/// in the table is its dense index: the bit rows of every other RTL fact use
/// it. Ascending order makes the index a stable rank, so a fact that lists
/// roots by index lists them in root order.
///
/// The set of these roots is the set T(F) of the function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootTable<D: Dialect> {
    /// The named roots, ascending and without duplicates.
    roots: Vec<D::Root>,
}

impl<D: Dialect> RootTable<D> {
    /// Collects the roots that the live blocks of `function` name.
    #[must_use]
    pub fn collect<F: FunctionView<D>>(function: &F) -> Self {
        let mut roots = Vec::new();
        for block in function.node_ids() {
            for &inst in function.instructions(block) {
                let record = function.record(inst);
                for statement in record.statements() {
                    match statement {
                        Statement::Read { source, .. } => roots.extend(selection_roots(source)),
                        Statement::Write { target, .. } => roots.extend(target_roots(target)),
                        Statement::Raise(raise) => {
                            roots.extend(raise.binds.iter().map(|&(root, _)| root));
                        }
                        _ => {}
                    }
                }
            }
        }
        roots.sort_unstable();
        roots.dedup();
        Self { roots }
    }

    /// Returns how many roots the function names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns whether the function names no root.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Returns the named roots in ascending order.
    #[must_use]
    pub fn roots(&self) -> &[D::Root] {
        &self.roots
    }

    /// Returns the root at dense `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not below [`Self::len`].
    #[must_use]
    pub fn root(&self, index: usize) -> D::Root {
        self.roots[index]
    }

    /// Returns the dense index of `root`, or `None` when the function does
    /// not name it.
    #[must_use]
    pub fn index_of(&self, root: D::Root) -> Option<usize> {
        self.roots.binary_search(&root).ok()
    }

    /// Returns the dense index of a root of the function that built the
    /// table.
    pub(super) fn index_of_named(&self, root: D::Root) -> usize {
        self.index_of(root)
            .expect("the root table was collected from this function")
    }
}

/// The roots that one selection or one write target touches.
pub(super) struct TouchedRoots<'a, D: Dialect> {
    /// A root that is still to come before the candidates.
    single: Option<D::Root>,
    /// The candidates whose roots are still to come.
    candidates: slice::Iter<'a, Place<D>>,
}

impl<D: Dialect> Iterator for TouchedRoots<'_, D> {
    type Item = D::Root;

    fn next(&mut self) -> Option<D::Root> {
        self.single
            .take()
            .or_else(|| self.candidates.next().map(|place| place.root))
    }
}

/// Returns the roots that a read of `selection` touches.
pub(super) fn selection_roots<D: Dialect, V: Copy>(
    selection: Selection<'_, D, V>,
) -> TouchedRoots<'_, D> {
    match selection {
        Selection::Direct(place) => single(place.root),
        Selection::FiniteIndexed { candidates, .. } => many(candidates),
        Selection::SparseIndexed { bank, .. } => single(bank),
    }
}

/// Returns the roots that a write to `target` touches.
pub(super) fn target_roots<D: Dialect, V: Copy>(
    target: WriteTarget<'_, D, V>,
) -> TouchedRoots<'_, D> {
    match target {
        WriteTarget::Root { root, .. } => single(root),
        WriteTarget::FiniteIndexed { candidates, .. } => many(candidates),
        WriteTarget::SparseIndexed { bank, .. } => single(bank),
    }
}

fn single<'a, D: Dialect>(root: D::Root) -> TouchedRoots<'a, D> {
    TouchedRoots {
        single: Some(root),
        candidates: [].iter(),
    }
}

fn many<D: Dialect>(candidates: &[Place<D>]) -> TouchedRoots<'_, D> {
    TouchedRoots {
        single: None,
        candidates: candidates.iter(),
    }
}
