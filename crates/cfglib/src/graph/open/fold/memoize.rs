//! The fold that folds each marked node once and hands its finished answer
//! to every later reach.

extern crate alloc;

use alloc::collections::BTreeMap;

use super::walk::Walk;
use super::{FoldEnter, OpenFold};

#[cfg(test)]
mod tests;

/// What the memoizing walk keeps for one open frame.
struct Memo<K> {
    /// The node's mark: the key its answer is stored under, and the name of
    /// the frame while it is open.
    mark: Option<K>,
    /// The stack position of the shallowest open frame that a refusal in
    /// this frame's subtree named, or `usize::MAX` for none.
    ///
    /// The frame's answer depends on where the walk entered a cycle exactly
    /// when this position lies below the frame's own: the refused node is an
    /// ancestor whose answer was still missing. A refusal of the frame itself
    /// closes a cycle that the frame contains whole, so it keeps the answer
    /// storable.
    reaches: usize,
}

/// Absorbs a child that was refused because the frame at stack position
/// `target` is open: the cycle that the child closes reaches down to
/// `target`.
fn refuse_open<F: OpenFold>(
    walk: &mut Walk<F, Memo<F::Mark>>,
    fold: &mut F,
    child: &F::Node,
    target: usize,
) {
    if let Some(top) = walk.stack.last_mut() {
        top.extra.reaches = top.extra.reaches.min(target);
    }
    walk.absorb(fold, child, None);
}

/// Folds the space reachable from `root` in post-order, folding each marked
/// node once: `answers` holds the finished answers by mark, and the walk
/// reads and extends it.
pub(super) fn run<F: OpenFold>(
    answers: &mut BTreeMap<F::Mark, Option<F::Value>>,
    fold: &mut F,
    root: F::Node,
    max_depth: Option<usize>,
) -> Option<F::Value>
where
    F::Value: Clone,
{
    let mark = fold.mark(&root);
    if let Some(mark) = &mark
        && let Some(answer) = answers.get(mark)
    {
        return answer.clone();
    }
    let mut walk = Walk::<F, Memo<F::Mark>>::new();
    // The stack positions of the open frames by mark. No two open frames
    // share a mark: a child whose mark belongs to an open frame is refused
    // before it opens a frame.
    let mut open: BTreeMap<F::Mark, usize> = BTreeMap::new();

    if let Some(mark) = &mark {
        walk.marks.insert(mark.clone());
    }
    match fold.enter(&root) {
        FoldEnter::Leaf(value) => {
            if let Some(mark) = mark {
                answers.insert(mark, value.clone());
            }
            return value;
        }
        FoldEnter::Fold {
            accumulator,
            marks: scope,
        } => {
            if let Some(mark) = &mark {
                open.insert(mark.clone(), 0);
            }
            let memo = Memo {
                mark,
                reaches: usize::MAX,
            };
            walk.push_frame(fold, root, accumulator, scope, 0, memo);
        }
    }

    loop {
        let Some(child) = walk.next_child() else {
            let Some((node, value, memo)) = walk.finish_top(fold) else {
                unreachable!("the walk returns once its root finishes");
            };
            let position = walk.stack.len();
            if let Some(mark) = memo.mark {
                open.remove(&mark);
                if memo.reaches >= position {
                    answers.insert(mark, value.clone());
                }
            }
            let Some(parent) = walk.stack.last_mut() else {
                return value;
            };
            parent.extra.reaches = parent.extra.reaches.min(memo.reaches);
            walk.absorb(fold, &node, value);
            continue;
        };

        if !walk.admit_child(max_depth) {
            walk.absorb(fold, &child, None);
            continue;
        }
        let mark = fold.mark(&child);
        if let Some(mark) = &mark {
            if let Some(answer) = answers.get(mark) {
                let answer = answer.clone();
                walk.absorb(fold, &child, answer);
                continue;
            }
            if let Some(&target) = open.get(mark) {
                refuse_open(&mut walk, fold, &child, target);
                continue;
            }
            // A finished node whose answer depended on a cycle, kept marked
            // by a persisting scope: the fold chose to prune this reach.
            if walk.marks.contains(mark) {
                walk.absorb(fold, &child, None);
                continue;
            }
        }

        let entry_marks = walk.marks.len();
        if let Some(mark) = &mark {
            walk.marks.insert(mark.clone());
        }
        match fold.enter(&child) {
            FoldEnter::Leaf(value) => {
                if let Some(mark) = mark {
                    answers.insert(mark, value.clone());
                }
                walk.absorb(fold, &child, value);
            }
            FoldEnter::Fold {
                accumulator,
                marks: scope,
            } => {
                if let Some(mark) = &mark {
                    open.insert(mark.clone(), walk.stack.len());
                }
                let memo = Memo {
                    mark,
                    reaches: usize::MAX,
                };
                walk.push_frame(fold, child, accumulator, scope, entry_marks, memo);
            }
        }
    }
}
