//! The fold that refuses a marked node it reaches again.

use super::walk::Walk;
use super::{FoldEnter, OpenFold};

/// Folds the space reachable from `root` in post-order, absorbing a child
/// whose mark is live, or that lies beyond `max_depth`, as `None`.
pub(super) fn run<F: OpenFold>(
    fold: &mut F,
    root: F::Node,
    max_depth: Option<usize>,
) -> Option<F::Value> {
    let mut walk = Walk::<F, ()>::new();

    let root_key = fold.mark(&root);
    if let Some(key) = root_key {
        walk.marks.insert(key);
    }
    match fold.enter(&root) {
        FoldEnter::Leaf(value) => return value,
        FoldEnter::Fold {
            accumulator,
            marks: scope,
        } => walk.push_frame(fold, root, accumulator, scope, 0, ()),
    }

    let mut result = None;
    while !walk.stack.is_empty() {
        let Some(child) = walk.next_child() else {
            let Some((node, value, ())) = walk.finish_top(fold) else {
                break;
            };
            if walk.stack.is_empty() {
                result = value;
            } else {
                walk.absorb(fold, &node, value);
            }
            continue;
        };

        let within_depth = walk.admit_child(max_depth);
        let child_key = fold.mark(&child);
        let refused = !within_depth
            || child_key
                .as_ref()
                .is_some_and(|key| walk.marks.contains(key));
        if refused {
            walk.absorb(fold, &child, None);
            continue;
        }

        let entry_marks = walk.marks.len();
        if let Some(key) = child_key {
            walk.marks.insert(key);
        }
        match fold.enter(&child) {
            FoldEnter::Leaf(value) => walk.absorb(fold, &child, value),
            FoldEnter::Fold {
                accumulator,
                marks: scope,
            } => walk.push_frame(fold, child, accumulator, scope, entry_marks, ()),
        }
    }
    result
}
