//! Golden-file comparison for the crate's output formats.
//!
//! A golden is the exact text one renderer produces for one case, stored
//! under `tests/golden/<area>/<case>.<format>` so a review reads the output
//! itself instead of a list of substrings. Set `UPDATE_GOLDEN=1` to write
//! the rendered text back and then review the diff.
//!
//! The integration tests include this file directly with `#[path]`: a test
//! binary cannot reach the crate's private test helpers, and one comparison
//! rule for every golden is worth more than the indirection costs.

use std::fs;
use std::path::{Path, PathBuf};

/// Compare `actual` against the golden at `tests/golden/<relative>`.
///
/// A golden file always ends in a newline, because the repository's
/// end-of-file hook adds one to every text file. A rendering without a final
/// newline, such as a JSON encoding, is therefore stored and compared with one
/// appended; the rendered bytes before it must match exactly.
///
/// # Panics
///
/// Panics when the golden is missing, unreadable, or differs from `actual`,
/// and when blessing cannot write the file.
pub(crate) fn assert_golden(relative: &str, actual: &str) {
    let path = golden_path(relative);
    let mut stored = std::string::String::from(actual);
    if !stored.ends_with('\n') {
        stored.push('\n');
    }
    if std::env::var_os("UPDATE_GOLDEN").is_some_and(|value| value == "1") {
        let parent = path.parent().expect("a golden path always has a directory");
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("creating {}: {error}", parent.display()));
        fs::write(&path, &stored)
            .unwrap_or_else(|error| panic!("blessing {}: {error}", path.display()));
        return;
    }

    let expected = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "reading golden {}: {error}\nre-run with UPDATE_GOLDEN=1 to create it",
            path.display()
        )
    });
    assert!(
        expected.replace("\r\n", "\n") == stored,
        "golden {relative} does not match; re-run with UPDATE_GOLDEN=1 to update it\n\
         --- expected ---\n{expected}\n--- actual ---\n{stored}"
    );
}

fn golden_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(relative)
}
