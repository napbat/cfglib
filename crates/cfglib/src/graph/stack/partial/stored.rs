//! Reading path summaries from their serialized form without trusting them.
//!
//! Replacing a file's summaries indexes the summary slots by every identity
//! the per-file table lists and the start table by each summary's start, and
//! stitching reads summaries through the start table, so a database whose
//! tables disagree with its summaries would panic or hide summaries in the
//! middle of an update or a query. Each type is therefore decoded into the
//! same columns with no behavior and becomes a summary, a set, or a database
//! only once it has been checked for what it can see of itself. Which graph
//! the identities belong to is not among those things.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{
    StackEdgeId, StackFileId, StackNodeId, StackPartialPath, StackPartialPathDatabase,
    StackPartialPathId, StackPartialPathSet, StackPartialPathStats,
};

/// The columns of a [`StackPartialPath`], exactly as they are written.
#[derive(serde::Deserialize)]
pub(super) struct Path {
    file: StackFileId,
    start: StackNodeId,
    end: StackNodeId,
    edges: Vec<StackEdgeId>,
}

impl TryFrom<Path> for StackPartialPath {
    type Error = &'static str;

    /// Accepts what extraction could have produced: a route of at least one
    /// edge, because a summary is recorded after the edge that reached its end,
    /// that takes no edge twice, because extraction cuts a route that would.
    fn try_from(stored: Path) -> Result<Self, Self::Error> {
        let Path {
            file,
            start,
            end,
            edges,
        } = stored;
        if edges.is_empty() {
            return Err("a path summary holds no edge");
        }
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("a path summary takes an edge twice");
        }
        Ok(Self {
            file,
            start,
            end,
            edges,
        })
    }
}

/// The columns of a [`StackPartialPathSet`], exactly as they are written.
#[derive(serde::Deserialize)]
pub(super) struct Set {
    file: StackFileId,
    paths: Vec<StackPartialPath>,
    stats: StackPartialPathStats,
}

impl TryFrom<Set> for StackPartialPathSet {
    type Error = &'static str;

    /// Accepts what `compute` could have produced: every summary extracted
    /// from the set's own file, and statistics that count them.
    fn try_from(stored: Set) -> Result<Self, Self::Error> {
        let Set { file, paths, stats } = stored;
        if paths.iter().any(|path| path.file() != file) {
            return Err("a path summary belongs to another file than its set");
        }
        if stats.partial_path_count != paths.len() {
            return Err("the statistics count other summaries than the set holds");
        }
        Ok(Self { file, paths, stats })
    }
}

/// The columns of a [`StackPartialPathDatabase`], exactly as they are written.
#[derive(serde::Deserialize)]
pub(super) struct Database {
    paths: Vec<Option<StackPartialPath>>,
    paths_by_start: Vec<Vec<StackPartialPathId>>,
    paths_by_file: Vec<Vec<StackPartialPathId>>,
}

impl TryFrom<Database> for StackPartialPathDatabase {
    type Error = &'static str;

    /// Accepts what `compute` and `replace_file` could have produced: each
    /// table has a row for every start or file of a live summary and lists, in
    /// installation order, exactly the live summaries that start or belong
    /// there. A replaced summary is a vacant slot that no row lists.
    fn try_from(stored: Database) -> Result<Self, Self::Error> {
        let Database {
            paths,
            paths_by_start,
            paths_by_file,
        } = stored;
        if grouped(&paths, paths_by_start.len(), |path| path.start().index())
            .is_none_or(|rows| rows != paths_by_start)
        {
            return Err("the per-start lists are not the live summaries grouped by their start");
        }
        if grouped(&paths, paths_by_file.len(), |path| path.file().index())
            .is_none_or(|rows| rows != paths_by_file)
        {
            return Err("the per-file lists are not the live summaries grouped by their file");
        }
        Ok(Self {
            paths,
            paths_by_start,
            paths_by_file,
        })
    }
}

/// The table a database keeps beside its summaries: `rows` lists, each holding
/// in identity order the live summaries `owner` assigns to it.
///
/// `None` when a live summary's owner has no row.
fn grouped(
    paths: &[Option<StackPartialPath>],
    rows: usize,
    owner: impl Fn(&StackPartialPath) -> usize,
) -> Option<Vec<Vec<StackPartialPathId>>> {
    let mut grouped = vec![Vec::new(); rows];
    for (index, path) in paths.iter().enumerate() {
        if let Some(path) = path {
            grouped
                .get_mut(owner(path))?
                .push(StackPartialPathId::from_index(index));
        }
    }
    Some(grouped)
}
