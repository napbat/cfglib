//! A sink that stops accepting text, the way a closed pipe or a full buffer
//! does.

use std::fmt;

/// Accepts at most `budget` bytes and then fails every further write.
pub struct FailingSink {
    budget: usize,
}

impl FailingSink {
    pub const fn with_budget(budget: usize) -> Self {
        Self { budget }
    }
}

impl fmt::Write for FailingSink {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let remaining = self.budget.checked_sub(text.len()).ok_or(fmt::Error)?;
        self.budget = remaining;
        Ok(())
    }
}

/// Every prefix budget shorter than `full` bytes must make `write` fail, and
/// any budget of at least `full` bytes must let it finish: a writer reports
/// exactly the failures its sink produces and swallows none.
pub fn assert_failure_is_reported_at_every_length(
    full: &str,
    write: impl Fn(&mut dyn fmt::Write) -> fmt::Result,
) {
    for budget in 0..full.len() {
        assert!(
            write(&mut FailingSink::with_budget(budget)).is_err(),
            "a sink that refuses the text after {budget} of {} bytes must fail the write",
            full.len()
        );
    }
    assert!(write(&mut FailingSink::with_budget(full.len())).is_ok());
}
