//! The frame stack that every revisit policy of the open fold walks with.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::open::marks::Marks;
use crate::graph::search::may_expand;

use super::{MarkScope, OpenFold};

/// The live mark set, with the order its marks were set in.
///
/// Isolation is an undo, not a copy: a node that sandboxes its children
/// records how many marks were set when it was entered, and taking back
/// everything its subtree added is dropping the marks past that length. A
/// mark is only ever set when absent, so the log and the set always hold the
/// same keys.
pub(super) struct UndoMarks<K> {
    set: Marks<K>,
    log: Vec<K>,
}

impl<K: Clone + Ord> UndoMarks<K> {
    fn new() -> Self {
        Self {
            set: Marks::new(),
            log: Vec::new(),
        }
    }

    pub(super) fn contains(&self, key: &K) -> bool {
        self.set.contains(key)
    }

    /// The number of marks set, which is also the position of the next one.
    pub(super) fn len(&self) -> usize {
        self.log.len()
    }

    /// Sets a mark the caller has checked is absent.
    pub(super) fn insert(&mut self, key: K) {
        self.set.insert(key.clone());
        self.log.push(key);
    }

    /// Takes back every mark set after the first `len`.
    fn truncate(&mut self, len: usize) {
        while self.log.len() > len {
            if let Some(key) = self.log.pop() {
                self.set.remove(&key);
            }
        }
    }
}

/// The marks an `Isolated` node restores, as lengths of the mark log.
struct Sandbox {
    /// The marks as seen at the node's entry, before its own key.
    entry: usize,
    /// The same marks plus the node's own key: what each child starts from.
    with_key: usize,
}

pub(super) struct Frame<F: OpenFold, X> {
    node: F::Node,
    accumulator: F::Accumulator,
    /// Where this node's children start on the walk's shared child stack.
    /// They are pushed in reverse, so the next one to fold is on top, and the
    /// frame has none left once the stack is back down to `base`.
    base: usize,
    sandbox: Option<Sandbox>,
    broke: bool,
    /// What one policy keeps for the frame beyond the shared bookkeeping.
    pub(super) extra: X,
}

/// The state of one walk: its frames, their children, and the live marks.
pub(super) struct Walk<F: OpenFold, X> {
    pub(super) stack: Vec<Frame<F, X>>,
    /// Every open frame's unfolded children, the deepest frame's on top.
    pending: Vec<F::Node>,
    /// The buffer `successors` fills, reused for every node.
    scratch: Vec<F::Node>,
    pub(super) marks: UndoMarks<F::Mark>,
}

impl<F: OpenFold, X> Walk<F, X> {
    pub(super) fn new() -> Self {
        Self {
            stack: Vec::new(),
            pending: Vec::new(),
            scratch: Vec::new(),
            marks: UndoMarks::new(),
        }
    }

    /// Opens a frame for `node`, whose own key (if it has one) is already the
    /// last mark set; `entry_marks` is how many marks were set before it.
    ///
    /// Inline: `open_fold_in` instantiates every policy's driver for a fold,
    /// and beside the other two the inliner left this out of line in the
    /// refusing walk, which made `open_fold_isolated_layered` 1.06x main.
    #[inline]
    pub(super) fn push_frame(
        &mut self,
        fold: &mut F,
        node: F::Node,
        accumulator: F::Accumulator,
        scope: MarkScope,
        entry_marks: usize,
        extra: X,
    ) {
        let base = self.pending.len();
        self.scratch.clear();
        fold.successors(&node, &mut self.scratch);
        self.pending.extend(self.scratch.drain(..).rev());
        // The sandbox restores the marks as seen at entry, before this node's
        // own key: once an isolated node finishes, a sibling may reach it
        // again through its own route.
        let sandbox = (scope == MarkScope::Isolated).then(|| Sandbox {
            entry: entry_marks,
            with_key: self.marks.len(),
        });
        self.stack.push(Frame {
            node,
            accumulator,
            base,
            sandbox,
            broke: false,
            extra,
        });
    }

    /// The top frame's next child, or `None` once the frame has no child
    /// left or has stopped early.
    pub(super) fn next_child(&mut self) -> Option<F::Node> {
        let frame = self.stack.last()?;
        if frame.broke || self.pending.len() <= frame.base {
            None
        } else {
            self.pending.pop()
        }
    }

    /// Pops the top frame and finishes it, returning its node, its answer
    /// and its policy state.
    pub(super) fn finish_top(&mut self, fold: &mut F) -> Option<(F::Node, Option<F::Value>, X)> {
        let frame = self.stack.pop()?;
        // A node stopped early leaves its remaining children unfolded.
        self.pending.truncate(frame.base);
        let value = fold.finish(&frame.node, frame.accumulator);
        if let Some(sandbox) = frame.sandbox {
            self.marks.truncate(sandbox.entry);
        }
        Some((frame.node, value, frame.extra))
    }

    /// Restores the marks that the top frame's next child starts from, and
    /// returns whether that child lies within the depth bound.
    pub(super) fn admit_child(&mut self, max_depth: Option<usize>) -> bool {
        let Some(frame) = self.stack.last() else {
            return false;
        };
        // An isolated parent resets its children's shared start state: marks
        // as seen at its own entry, plus its own key. For its first child
        // that is the state it already has.
        if let Some(sandbox) = &frame.sandbox {
            self.marks.truncate(sandbox.with_key);
        }
        // The parent is the top frame, at depth `stack.len() - 1`; its child
        // is refused exactly when that parent may not expand.
        may_expand(max_depth, self.stack.len() - 1)
    }

    /// Merges one child's answer into the top frame, which stops early when
    /// the fold breaks.
    pub(super) fn absorb(&mut self, fold: &mut F, child: &F::Node, value: Option<F::Value>) {
        if let Some(parent) = self.stack.last_mut()
            && fold
                .absorb(&mut parent.accumulator, child, value)
                .is_break()
        {
            parent.broke = true;
        }
    }
}
