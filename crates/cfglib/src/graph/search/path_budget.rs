//! The bounds of a bounded path enumeration, the record of which of them
//! refused work, and the statistics that report it.

use super::SearchOrder;

/// The frontier order and the bounds of a bounded path enumeration.
///
/// One config serves every path enumeration of name resolution: scope
/// resolution ([`ScopeResolution`](crate::ScopeResolution),
/// [`ScopeFirstResolution`](crate::ScopeFirstResolution) and
/// [`ScopeEnvironmentCache`](crate::ScopeEnvironmentCache)), stack-graph
/// search and stitching ([`StackResolution`](crate::StackResolution)), and
/// partial-path extraction ([`StackPartialPathSet`](crate::StackPartialPathSet)).
/// A plain struct: write the fields that differ from the default (an
/// unbounded breadth-first enumeration) and take the rest with
/// `..PathSearchConfig::default()`.
///
/// Every resolver documents the fields it does not read. Scope resolution is
/// breadth-first by contract and keeps every candidate, so it reads neither
/// `order` nor `max_complete_path_count`; partial-path extraction keeps every
/// summary, so it does not read `max_complete_path_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathSearchConfig {
    /// The order in which queued path prefixes are expanded.
    pub order: SearchOrder,
    /// The most edges a path may have. `None` is unbounded.
    pub max_path_length: Option<usize>,
    /// The most path prefixes to expand. `None` is unbounded.
    pub max_path_count: Option<usize>,
    /// The most complete paths to retain before shadow filtering. `None` is
    /// unbounded.
    pub max_complete_path_count: Option<usize>,
}

impl Default for PathSearchConfig {
    /// An unbounded breadth-first enumeration.
    fn default() -> Self {
        Self {
            order: SearchOrder::BreadthFirst,
            max_path_length: None,
            max_path_count: None,
            max_complete_path_count: None,
        }
    }
}

/// The work of one bounded path enumeration and the bounds that refused some
/// of it.
///
/// A truncation flag is set only by an actual refusal, never by merely
/// reaching a bound, so a flag means "a binding may have been omitted".
/// Every enumeration counts in these terms, which keeps their statistics
/// comparable; a resolver that counts a field its own way says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PathSearchStats {
    /// The path prefixes expanded.
    pub expanded_path_count: usize,
    /// The complete paths found, before shadow filtering and the retention
    /// bound.
    pub complete_path_count: usize,
    /// Whether an otherwise admissible extension exceeded the length bound.
    pub path_length_limited: bool,
    /// Whether the count bound refused a queued prefix.
    pub path_count_limited: bool,
    /// Whether the retention bound refused a complete path.
    pub complete_path_count_limited: bool,
}

impl PathSearchStats {
    /// Whether a bound may have omitted a binding.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.path_length_limited || self.path_count_limited || self.complete_path_count_limited
    }
}

/// Counts a path enumeration's work against its configured bounds and records
/// which bound turned work away.
///
/// Every enumeration (direct, linear, stitched, and structural) asks the same
/// three questions in the same terms, so that their [`PathSearchStats`] are
/// comparable.
pub(crate) struct PathBudget {
    max_path_length: Option<usize>,
    max_path_count: Option<usize>,
    max_complete_path_count: Option<usize>,
    stats: PathSearchStats,
}

impl PathBudget {
    /// A budget with the bounds of `config` and no work counted yet.
    pub(crate) const fn new(config: &PathSearchConfig) -> Self {
        Self {
            max_path_length: config.max_path_length,
            max_path_count: config.max_path_count,
            max_complete_path_count: config.max_complete_path_count,
            stats: PathSearchStats {
                expanded_path_count: 0,
                complete_path_count: 0,
                path_length_limited: false,
                path_count_limited: false,
                complete_path_count_limited: false,
            },
        }
    }

    /// Count one more expanded prefix, or return `false` and record the
    /// path-count truncation when the bound leaves no room for it.
    pub(crate) fn admit_expansion(&mut self) -> bool {
        if self
            .max_path_count
            .is_some_and(|limit| self.stats.expanded_path_count >= limit)
        {
            self.stats.path_count_limited = true;
            return false;
        }
        self.stats.expanded_path_count += 1;
        true
    }

    /// Whether an extension of `extended_len` stored edges fits the length
    /// bound; otherwise record the path-length truncation and return `false`.
    pub(crate) fn admit_length(&mut self, extended_len: usize) -> bool {
        if self
            .max_path_length
            .is_some_and(|limit| extended_len > limit)
        {
            self.stats.path_length_limited = true;
            return false;
        }
        true
    }

    /// Count one complete path found, and whether it is retained given that
    /// `retained` complete paths already are; otherwise record the
    /// complete-path truncation and return `false`.
    pub(crate) fn admit_complete(&mut self, retained: usize) -> bool {
        self.stats.complete_path_count += 1;
        if self
            .max_complete_path_count
            .is_none_or(|limit| retained < limit)
        {
            return true;
        }
        self.stats.complete_path_count_limited = true;
        false
    }

    /// The work counted so far and the bounds that refused some of it.
    pub(crate) const fn stats(&self) -> PathSearchStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(
        max_path_length: Option<usize>,
        max_path_count: Option<usize>,
        max_complete_path_count: Option<usize>,
    ) -> PathBudget {
        PathBudget::new(&PathSearchConfig {
            max_path_length,
            max_path_count,
            max_complete_path_count,
            ..PathSearchConfig::default()
        })
    }

    #[test]
    fn an_unbounded_budget_admits_everything_and_never_reports_truncation() {
        let mut budget = budget(None, None, None);
        for retained in 0..100 {
            assert!(budget.admit_expansion());
            assert!(budget.admit_length(usize::MAX));
            assert!(budget.admit_complete(retained));
        }
        let stats = budget.stats();
        assert_eq!(stats.expanded_path_count, 100);
        assert_eq!(stats.complete_path_count, 100);
        assert!(!stats.is_truncated());
    }

    #[test]
    fn the_count_bound_refuses_the_prefix_after_the_last_admitted_one() {
        let mut budget = budget(None, Some(2), None);
        assert!(budget.admit_expansion());
        assert!(budget.admit_expansion());
        assert!(
            !budget.stats().path_count_limited,
            "reaching the bound is no loss"
        );
        assert!(!budget.admit_expansion());
        assert!(budget.stats().path_count_limited);
        assert_eq!(budget.stats().expanded_path_count, 2);
    }

    #[test]
    fn the_length_bound_admits_exactly_the_paths_of_that_many_edges() {
        let mut budget = budget(Some(3), None, None);
        assert!(budget.admit_length(3));
        assert!(!budget.stats().path_length_limited);
        assert!(!budget.admit_length(4));
        assert!(budget.stats().path_length_limited);
    }

    #[test]
    fn the_complete_bound_counts_every_path_but_retains_only_the_first_few() {
        let mut budget = budget(None, None, Some(1));
        assert!(budget.admit_complete(0));
        assert!(!budget.stats().complete_path_count_limited);
        assert!(!budget.admit_complete(1));
        assert!(budget.stats().complete_path_count_limited);
        assert_eq!(budget.stats().complete_path_count, 2);
    }

    #[test]
    fn the_default_config_is_an_unbounded_breadth_first_enumeration() {
        let config = PathSearchConfig::default();
        assert_eq!(config.order, SearchOrder::BreadthFirst);
        assert_eq!(
            (
                config.max_path_length,
                config.max_path_count,
                config.max_complete_path_count
            ),
            (None, None, None)
        );
    }
}
