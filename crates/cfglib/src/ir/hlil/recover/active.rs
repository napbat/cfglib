//! The regions under construction, and the variables that hold what they
//! entered.
//!
//! A statement of a region body concerns only the regions whose entered
//! object it can name. Each variable therefore lists the open regions that
//! have it among their aliases, so that a statement costs the regions that it
//! concerns, and not every region that is open.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::collections::btree_map::{BTreeMap, Entry};
use alloc::vec::Vec;

/// One region under construction: its enter operation, the variables that
/// currently hold the entered object (grown by copies and shrunk by
/// reassignment, in program order), and the exit statements that its
/// construct re-owns.
pub(super) struct ActiveRegion<Op, V, S> {
    pub(super) enter: Op,
    pub(super) aliases: BTreeSet<V>,
    pub(super) exits: Vec<S>,
}

/// The open regions, the innermost last.
pub(super) struct ActiveRegions<Op, V, S> {
    regions: Vec<ActiveRegion<Op, V, S>>,
    /// For each variable, the positions in `regions`, ascending, of the
    /// regions that have it among their aliases.
    holders: BTreeMap<V, Vec<usize>>,
}

impl<Op, V: Copy + Ord, S> ActiveRegions<Op, V, S> {
    pub(super) const fn new() -> Self {
        Self {
            regions: Vec::new(),
            holders: BTreeMap::new(),
        }
    }

    pub(super) const fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Opens a region over the object that `entered` holds, which `released`
    /// is the view of that a release names.
    pub(super) fn push(&mut self, enter: Op, entered: V, released: V) {
        let position = self.regions.len();
        let aliases = BTreeSet::from([entered, released]);
        for &variable in &aliases {
            self.holders.entry(variable).or_default().push(position);
        }
        self.regions.push(ActiveRegion {
            enter,
            aliases,
            exits: Vec::new(),
        });
    }

    /// Closes the innermost region.
    pub(super) fn pop(&mut self) -> Option<ActiveRegion<Op, V, S>> {
        let region = self.regions.pop()?;
        for &variable in &region.aliases {
            if let Entry::Occupied(mut holders) = self.holders.entry(variable) {
                holders.get_mut().pop();
                if holders.get().is_empty() {
                    holders.remove();
                }
            }
        }
        Some(region)
    }

    /// The innermost region that has `released` among its aliases and whose
    /// enter operation satisfies `accepts`, which asks the dialect whether
    /// the operation of the statement releases it. The cost is one call for
    /// each region that holds `released`, innermost first, up to the match.
    pub(super) fn innermost_releasing(
        &self,
        released: V,
        accepts: impl Fn(&Op) -> bool,
    ) -> Option<usize> {
        self.holders
            .get(&released)?
            .iter()
            .rev()
            .copied()
            .find(|&position| accepts(&self.regions[position].enter))
    }

    /// Records `exit` as an exit that the region at `position` re-owns.
    pub(super) fn add_exit(&mut self, position: usize, exit: S) {
        self.regions[position].exits.push(exit);
    }

    /// Follows an assignment of `target` through the aliases: a region that
    /// has `source` among its aliases gains `target`, and every other region
    /// loses it.
    pub(super) fn assign(&mut self, target: V, source: Option<V>) {
        let gained = source.and_then(|source| self.holders.get(&source).cloned());
        let before = match gained {
            Some(positions) if !positions.is_empty() => self.holders.insert(target, positions),
            _ => self.holders.remove(&target),
        }
        .unwrap_or_default();
        let after = self.holders.get(&target).map_or(&[][..], Vec::as_slice);
        for &position in &before {
            if after.binary_search(&position).is_err() {
                self.regions[position].aliases.remove(&target);
            }
        }
        for &position in after {
            if before.binary_search(&position).is_err() {
                self.regions[position].aliases.insert(target);
            }
        }
    }
}
