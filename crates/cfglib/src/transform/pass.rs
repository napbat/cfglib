//! Ordered composition of named transformation passes.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::fmt;

/// Stable, human-readable identity of a transformation pass.
///
/// Consumers should namespace identifiers that may appear in shared reports,
/// for example `"java.recover-loops"`. The pipeline deliberately permits the
/// same identifier more than once so callers can express repeated or
/// fixpoint-style schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassId(&'static str);

impl PassId {
    /// Creates an identifier from a process-lifetime name.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Returns the stable name carried by this identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for PassId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Whether one pass changed its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PassChange {
    /// The target was unchanged.
    #[default]
    Unchanged,
    /// The target was changed.
    Changed,
}

impl PassChange {
    /// Converts a boolean change predicate into a pass result.
    #[must_use]
    pub const fn from_changed(changed: bool) -> Self {
        if changed {
            Self::Changed
        } else {
            Self::Unchanged
        }
    }

    /// Returns whether the pass changed its target.
    #[must_use]
    pub const fn is_changed(self) -> bool {
        matches!(self, Self::Changed)
    }
}

/// One named transformation over `T`.
///
/// A pass may keep analyses or configuration in its own value. Pipelines call
/// passes mutably and therefore support stateful passes without imposing an
/// analysis-cache model on the target. On failure, any mutation already made
/// by the failing pass remains visible; transactional behavior belongs in a
/// pass that can clone or otherwise stage its own target.
pub trait Pass<T> {
    /// Consumer-defined failure type shared by one pipeline.
    type Error;

    /// Returns the stable identity used in schedules and reports.
    fn id(&self) -> PassId;

    /// Applies this pass to `target`.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the transformation cannot complete.
    fn run(&mut self, target: &mut T) -> Result<PassChange, Self::Error>;
}

/// A named pass backed by a closure.
pub struct PassFn<F> {
    id: PassId,
    run: F,
}

/// Creates a named pass from a fallible transformation closure.
#[must_use]
pub const fn pass_fn<F>(id: PassId, run: F) -> PassFn<F> {
    PassFn { id, run }
}

impl<T, E, F> Pass<T> for PassFn<F>
where
    F: FnMut(&mut T) -> Result<PassChange, E>,
{
    type Error = E;

    fn id(&self) -> PassId {
        self.id
    }

    fn run(&mut self, target: &mut T) -> Result<PassChange, Self::Error> {
        (self.run)(target)
    }
}

/// Outcome recorded for one successful pass execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassExecution {
    id: PassId,
    change: PassChange,
}

impl PassExecution {
    /// Returns the executed pass identity.
    #[must_use]
    pub const fn id(self) -> PassId {
        self.id
    }

    /// Returns whether this execution changed the target.
    #[must_use]
    pub const fn change(self) -> PassChange {
        self.change
    }
}

/// Ordered outcomes from the successfully completed part of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[must_use]
pub struct PassReport {
    executions: Vec<PassExecution>,
}

impl PassReport {
    /// Returns executions in schedule order.
    #[must_use]
    pub fn executions(&self) -> &[PassExecution] {
        &self.executions
    }

    /// Returns the number of completed pass executions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.executions.len()
    }

    /// Returns whether no pass completed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.executions.is_empty()
    }

    /// Returns the number of executions that changed the target.
    #[must_use]
    pub fn changed_count(&self) -> usize {
        self.executions
            .iter()
            .filter(|execution| execution.change.is_changed())
            .count()
    }

    /// Returns whether at least one completed pass changed the target.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        self.executions
            .iter()
            .any(|execution| execution.change.is_changed())
    }
}

/// Failure of one named pass after zero or more successful executions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassFailure<E> {
    pass: PassId,
    error: E,
    completed: PassReport,
}

impl<E> PassFailure<E> {
    /// Returns the identity of the pass that failed.
    #[must_use]
    pub const fn pass(&self) -> PassId {
        self.pass
    }

    /// Borrows the consumer-defined error.
    #[must_use]
    pub const fn error(&self) -> &E {
        &self.error
    }

    /// Returns the report for passes completed before the failure.
    pub const fn completed(&self) -> &PassReport {
        &self.completed
    }

    /// Splits this failure into its pass identity, error, and partial report.
    pub fn into_parts(self) -> (PassId, E, PassReport) {
        (self.pass, self.error, self.completed)
    }
}

impl<E: fmt::Display> fmt::Display for PassFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "pass `{}` failed: {}", self.pass, self.error)
    }
}

impl<E: core::error::Error + 'static> core::error::Error for PassFailure<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Mutable, insertion-ordered schedule of transformation passes.
///
/// Every pass in one pipeline shares a target and error type. The pipeline
/// stops at the first error and reports the successfully completed prefix. It
/// does not roll back prior passes or mutation made by the failing pass.
///
/// # Examples
///
/// ```
/// use core::convert::Infallible;
/// use cfglib::{PassChange, PassId, PassPipeline, pass_fn};
///
/// let mut pipeline = PassPipeline::<usize, Infallible>::new();
/// pipeline.push(pass_fn(PassId::new("example.increment"), |value: &mut usize| {
///     *value += 1;
///     Ok(PassChange::Changed)
/// }));
///
/// let mut value = 0;
/// let report = pipeline.run_infallible(&mut value);
/// assert_eq!(value, 1);
/// assert_eq!(report.changed_count(), 1);
/// ```
pub struct PassPipeline<'pass, T, E> {
    passes: Vec<Box<dyn Pass<T, Error = E> + 'pass>>,
}

impl<'pass, T, E> PassPipeline<'pass, T, E> {
    /// Creates an empty schedule.
    #[must_use]
    pub const fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Appends a pass to the schedule.
    pub fn push<P>(&mut self, pass: P)
    where
        P: Pass<T, Error = E> + 'pass,
    {
        self.passes.push(Box::new(pass));
    }

    /// Appends a pass and returns the updated schedule.
    #[must_use]
    pub fn with_pass<P>(mut self, pass: P) -> Self
    where
        P: Pass<T, Error = E> + 'pass,
    {
        self.push(pass);
        self
    }

    /// Returns scheduled pass identities in execution order.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = PassId> + '_ {
        self.passes.iter().map(|pass| pass.id())
    }

    /// Returns the number of scheduled executions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Returns whether the schedule is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Executes every pass in insertion order.
    ///
    /// # Errors
    ///
    /// Returns the failed pass identity, its error, and the completed-prefix
    /// report. Mutations are not rolled back.
    pub fn run(&mut self, target: &mut T) -> Result<PassReport, PassFailure<E>> {
        let mut report = PassReport::default();
        for pass in &mut self.passes {
            let id = pass.id();
            match pass.run(target) {
                Ok(change) => report.executions.push(PassExecution { id, change }),
                Err(error) => {
                    return Err(PassFailure {
                        pass: id,
                        error,
                        completed: report,
                    });
                }
            }
        }
        Ok(report)
    }
}

impl<T, E> Default for PassPipeline<'_, T, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PassPipeline<'_, T, Infallible> {
    /// Executes an infallible schedule in insertion order.
    pub fn run_infallible(&mut self, target: &mut T) -> PassReport {
        match self.run(target) {
            Ok(report) => report,
            Err(PassFailure { error, .. }) => match error {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn runs_in_order_and_reports_changes() {
        let mut pipeline = PassPipeline::<Vec<&str>, Infallible>::new();
        pipeline.push(pass_fn(PassId::new("first"), |target: &mut Vec<&str>| {
            target.push("first");
            Ok(PassChange::Changed)
        }));
        pipeline.push(pass_fn(
            PassId::new("observe"),
            |_target: &mut Vec<&str>| Ok(PassChange::Unchanged),
        ));
        pipeline.push(pass_fn(PassId::new("last"), |target: &mut Vec<&str>| {
            target.push("last");
            Ok(PassChange::Changed)
        }));

        assert_eq!(
            pipeline.ids().collect::<Vec<_>>(),
            vec![
                PassId::new("first"),
                PassId::new("observe"),
                PassId::new("last"),
            ]
        );
        let mut target = Vec::new();
        let report = pipeline.run_infallible(&mut target);

        assert_eq!(target, vec!["first", "last"]);
        assert_eq!(report.len(), 3);
        assert_eq!(report.changed_count(), 2);
        assert!(report.is_changed());
        assert_eq!(report.executions()[1].change(), PassChange::Unchanged);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestError;

    impl fmt::Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("stopped")
        }
    }

    impl core::error::Error for TestError {}

    #[test]
    fn attributes_failure_and_retains_completed_prefix() {
        let mut pipeline = PassPipeline::<usize, TestError>::new();
        pipeline.push(pass_fn(PassId::new("increment"), |target: &mut usize| {
            *target += 1;
            Ok(PassChange::Changed)
        }));
        pipeline.push(pass_fn(PassId::new("fail"), |target: &mut usize| {
            *target += 10;
            Err(TestError)
        }));
        pipeline.push(pass_fn(PassId::new("unreached"), |target: &mut usize| {
            *target += 100;
            Ok(PassChange::Changed)
        }));

        let mut target = 0;
        let failure = pipeline.run(&mut target).expect_err("second pass fails");

        assert_eq!(target, 11);
        assert_eq!(failure.pass(), PassId::new("fail"));
        assert_eq!(failure.error(), &TestError);
        assert_eq!(failure.completed().len(), 1);
        assert_eq!(failure.to_string(), "pass `fail` failed: stopped");
    }

    #[test]
    fn empty_pipeline_has_an_empty_report() {
        let mut pipeline = PassPipeline::<(), Infallible>::new();
        let report = pipeline.run_infallible(&mut ());
        assert!(pipeline.is_empty());
        assert!(report.is_empty());
        assert!(!report.is_changed());
    }
}
