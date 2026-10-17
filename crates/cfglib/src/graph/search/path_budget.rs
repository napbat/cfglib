//! The resource bounds of a bounded path enumeration and the record of which
//! of them refused work.

/// Counts a path enumeration's work against its configured bounds and records
/// which bound turned work away.
///
/// A truncation flag is set only by an actual refusal, never by merely reaching
/// a bound, so a flag means "a binding may have been omitted". Every
/// enumeration (direct, linear, stitched, and structural) asks the same three
/// questions in the same terms, which is what keeps their statistics
/// comparable.
pub(crate) struct PathBudget {
    max_path_length: Option<usize>,
    max_path_count: Option<usize>,
    max_complete_path_count: Option<usize>,
    expanded_path_count: usize,
    complete_path_count: usize,
    path_length_limited: bool,
    path_count_limited: bool,
    complete_path_count_limited: bool,
}

impl PathBudget {
    /// A budget with the given bounds and no bound on retained complete paths.
    /// `None` leaves a bound unlimited.
    pub(crate) const fn new(max_path_length: Option<usize>, max_path_count: Option<usize>) -> Self {
        Self {
            max_path_length,
            max_path_count,
            max_complete_path_count: None,
            expanded_path_count: 0,
            complete_path_count: 0,
            path_length_limited: false,
            path_count_limited: false,
            complete_path_count_limited: false,
        }
    }

    /// This budget with a bound on the complete paths retained.
    pub(crate) const fn with_max_complete_path_count(
        mut self,
        max_complete_path_count: Option<usize>,
    ) -> Self {
        self.max_complete_path_count = max_complete_path_count;
        self
    }

    /// Count one more expanded prefix, or return `false` and record the
    /// path-count truncation when the bound leaves no room for it.
    pub(crate) fn admit_expansion(&mut self) -> bool {
        if self
            .max_path_count
            .is_some_and(|limit| self.expanded_path_count >= limit)
        {
            self.path_count_limited = true;
            return false;
        }
        self.expanded_path_count += 1;
        true
    }

    /// Whether an extension of `extended_len` stored edges fits the length
    /// bound; otherwise record the path-length truncation and return `false`.
    pub(crate) fn admit_length(&mut self, extended_len: usize) -> bool {
        if self
            .max_path_length
            .is_some_and(|limit| extended_len > limit)
        {
            self.path_length_limited = true;
            return false;
        }
        true
    }

    /// Count one complete path found, and whether it is retained given that
    /// `retained` complete paths already are; otherwise record the
    /// complete-path truncation and return `false`.
    pub(crate) fn admit_complete(&mut self, retained: usize) -> bool {
        self.complete_path_count += 1;
        if self
            .max_complete_path_count
            .is_none_or(|limit| retained < limit)
        {
            return true;
        }
        self.complete_path_count_limited = true;
        false
    }

    /// Prefixes admitted for expansion so far.
    pub(crate) const fn expanded_path_count(&self) -> usize {
        self.expanded_path_count
    }

    /// Complete paths found so far, retained or not.
    pub(crate) const fn complete_path_count(&self) -> usize {
        self.complete_path_count
    }

    /// Whether an extension was refused for exceeding the length bound.
    pub(crate) const fn path_length_limited(&self) -> bool {
        self.path_length_limited
    }

    /// Whether the path-count bound refused a queued prefix.
    pub(crate) const fn path_count_limited(&self) -> bool {
        self.path_count_limited
    }

    /// Whether a complete path was refused retention.
    pub(crate) const fn complete_path_count_limited(&self) -> bool {
        self.complete_path_count_limited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unbounded_budget_admits_everything_and_never_reports_truncation() {
        let mut budget = PathBudget::new(None, None);
        for retained in 0..100 {
            assert!(budget.admit_expansion());
            assert!(budget.admit_length(usize::MAX));
            assert!(budget.admit_complete(retained));
        }
        assert_eq!(budget.expanded_path_count(), 100);
        assert_eq!(budget.complete_path_count(), 100);
        assert!(
            !budget.path_length_limited()
                && !budget.path_count_limited()
                && !budget.complete_path_count_limited()
        );
    }

    #[test]
    fn the_count_bound_refuses_the_prefix_after_the_last_admitted_one() {
        let mut budget = PathBudget::new(None, Some(2));
        assert!(budget.admit_expansion());
        assert!(budget.admit_expansion());
        assert!(
            !budget.path_count_limited(),
            "reaching the bound is no loss"
        );
        assert!(!budget.admit_expansion());
        assert!(budget.path_count_limited());
        assert_eq!(budget.expanded_path_count(), 2);
    }

    #[test]
    fn the_length_bound_admits_exactly_the_paths_of_that_many_edges() {
        let mut budget = PathBudget::new(Some(3), None);
        assert!(budget.admit_length(3));
        assert!(!budget.path_length_limited());
        assert!(!budget.admit_length(4));
        assert!(budget.path_length_limited());
    }

    #[test]
    fn the_complete_bound_counts_every_path_but_retains_only_the_first_few() {
        let mut budget = PathBudget::new(None, None).with_max_complete_path_count(Some(1));
        assert!(budget.admit_complete(0));
        assert!(!budget.complete_path_count_limited());
        assert!(!budget.admit_complete(1));
        assert!(budget.complete_path_count_limited());
        assert_eq!(budget.complete_path_count(), 2);
    }
}
