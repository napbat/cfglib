//! Explicit may-alias equivalence classes.
//!
//! Consumers can populate [`AliasSets`] from language-specific points-to or
//! binding information and pass it directly to [`MemorySSA`](crate::MemorySSA).

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::dataflow::VariableId;
use crate::dataflow::memory::MemoryAlias;
use crate::union_find::DisjointSet;

/// Caller-populated may-alias equivalence classes.
///
/// Equal locations always alias. Unequal locations alias only after they have
/// been joined with [`AliasSets::merge`], directly or transitively. Passing an
/// instance to [`MemorySSA`](crate::MemorySSA) therefore promises that every
/// unmerged pair is disjoint.
#[derive(Debug, Clone)]
pub struct AliasSets<V> {
    sets: DisjointSet,
    variable_to_id: BTreeMap<V, usize>,
    id_to_variable: Vec<V>,
}

impl<V: VariableId> AliasSets<V> {
    /// Create empty alias sets.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sets: DisjointSet::default(),
            variable_to_id: BTreeMap::new(),
            id_to_variable: Vec::new(),
        }
    }

    fn get_or_insert(&mut self, variable: V) -> usize {
        if let Some(&id) = self.variable_to_id.get(&variable) {
            return id;
        }
        let id = self.sets.push();
        self.variable_to_id.insert(variable.clone(), id);
        self.id_to_variable.push(variable);
        id
    }

    /// Registers one location as a singleton alias set.
    pub fn insert(&mut self, variable: V) {
        self.get_or_insert(variable);
    }

    /// Checks whether two locations are in the same may-alias class.
    #[must_use]
    pub fn may_alias(&self, left: &V, right: &V) -> bool {
        if left == right {
            return true;
        }
        let Some(&left_id) = self.variable_to_id.get(left) else {
            return false;
        };
        let Some(&right_id) = self.variable_to_id.get(right) else {
            return false;
        };
        self.sets.root(left_id) == self.sets.root(right_id)
    }

    /// Gets the alias-set representative for a registered location.
    #[must_use]
    pub fn alias_set(&self, variable: &V) -> Option<&V> {
        let &id = self.variable_to_id.get(variable)?;
        self.id_to_variable.get(self.sets.root(id))
    }

    /// Merge two variables into the same alias set.
    pub fn merge(&mut self, left: V, right: V) {
        let left_id = self.get_or_insert(left);
        let right_id = self.get_or_insert(right);
        self.sets.union(left_id, right_id);
    }

    /// Returns the number of registered alias sets.
    #[must_use]
    pub fn set_count(&self) -> usize {
        let mut roots = alloc::collections::BTreeSet::new();
        for id in 0..self.sets.len() {
            roots.insert(self.sets.root(id));
        }
        roots.len()
    }
}

impl<V: VariableId> Default for AliasSets<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: VariableId> MemoryAlias<V> for AliasSets<V> {
    fn may_alias(&self, left: &V, right: &V) -> bool {
        Self::may_alias(self, left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_creates_alias() {
        let mut sets = AliasSets::new();
        let a = 0_u16;
        let b = 1_u16;
        sets.merge(a, b);
        assert!(sets.may_alias(&a, &b));
    }

    #[test]
    fn unrelated_not_aliased() {
        let mut sets = AliasSets::new();
        sets.insert(0_u16);
        sets.insert(1_u16);
        assert!(!sets.may_alias(&0, &1));
    }

    #[test]
    fn transitive_alias() {
        let mut sets = AliasSets::new();
        sets.merge(0_u16, 1);
        sets.merge(1, 2);
        assert!(sets.may_alias(&0, &2));
    }

    #[test]
    fn set_count_drops_once_per_effective_merge() {
        let mut sets = AliasSets::new();
        sets.insert(0_u16);
        sets.insert(1_u16);
        sets.insert(2_u16);
        assert_eq!(sets.set_count(), 3);
        sets.merge(0, 1);
        assert_eq!(sets.set_count(), 2);
        sets.merge(1, 0);
        assert_eq!(sets.set_count(), 2, "the pair is already one set");
        sets.merge(2, 7);
        assert_eq!(sets.set_count(), 2, "7 is new and joins the set of 2");
    }

    #[test]
    fn merged_locations_share_one_representative() {
        let mut sets = AliasSets::new();
        sets.merge(0_u16, 1);
        sets.merge(1, 2);
        sets.insert(9);
        assert_eq!(sets.alias_set(&0), sets.alias_set(&2));
        assert_ne!(sets.alias_set(&0), sets.alias_set(&9));
        assert_eq!(sets.alias_set(&5), None);
    }
}
