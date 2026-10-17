//! Post-order folding over an open node space.
//!
//! A search can only stop; a fold computes each node's answer from its
//! children's answers on the unwind. [`open_fold_post_order`] owns the frame
//! bookkeeping that every hand-written fold repeats — the accumulator stack,
//! the cycle guard, the depth bound — while the [`OpenFold`] implementation
//! keeps full control of the three things real folds disagree on:
//!
//! - **What answers and how they combine.** [`OpenFold::enter`] may answer as
//!   a leaf without descending (a declaration that hides everything below
//!   it), or open an accumulator; [`OpenFold::absorb`] merges each child's
//!   answer — first-match, all-must-agree, sticky-ambiguity — and may stop
//!   the node's remaining children early; [`OpenFold::finish`] turns the
//!   accumulator into the node's own answer, rewriting it on the way up.
//! - **Which nodes guard against revisiting.** [`OpenFold::mark`] returns an
//!   orderable key for exactly the nodes that participate in the cycle
//!   guard; nodes without a key are folded every time they are reached. The
//!   node type itself needs no ordering, so evaluators over computed values
//!   (a type algebra, a substituted term) fold directly.
//! - **How far a mark reaches.** Keys accumulate in one live set; a
//!   [`MarkScope::Isolated`] node sandboxes its children, so each child's
//!   subtree starts from the marks as seen at that node and leaves nothing
//!   behind — including the node's own key once it finishes. Choosing
//!   `Isolated` at every node yields exact per-path marking (a diamond's
//!   shared base is folded once per route — the re-fold that detects
//!   ambiguity), while [`MarkScope::Shared`] keeps marks for the rest of the
//!   walk (an evaluator that prunes every re-reached reference).
//!
//! A refused child — already marked, or beyond the depth bound — is absorbed
//! as `None`: it contributes exactly the nothing a missing subtree
//! contributes, so accumulators observe every child exactly once.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::marks::Marks;

/// Bounds for one [`open_fold_post_order`] walk.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenFoldConfig {
    max_depth: Option<usize>,
}

impl OpenFoldConfig {
    /// Creates the unbounded configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_depth: None }
    }

    /// Refuses children deeper than `depth` levels below the root.
    ///
    /// The root is depth zero. A refused child is absorbed as `None`, the
    /// same contribution a marked child makes.
    #[must_use]
    pub const fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
}

/// How far the marks written inside one folded node's subtree reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkScope {
    /// Children share the live mark set and their marks persist for the
    /// rest of the walk (until discarded by an `Isolated` ancestor).
    Shared,
    /// Each child's subtree starts from the marks as seen at this node and
    /// leaves nothing behind; the node's own key is also discarded once it
    /// finishes. Choosing this at every node yields exact per-path marking.
    Isolated,
}

/// What [`OpenFold::enter`] decides for one reached node.
#[derive(Debug)]
pub enum FoldEnter<T, C> {
    /// Answer immediately without visiting children.
    ///
    /// The node is still marked (per [`OpenFold::mark`]), so a later reach
    /// under a persisting scope is refused rather than re-answered.
    Leaf(Option<T>),
    /// Descend: fold every child into `accumulator`, guarded per `marks`.
    Fold {
        /// The empty combination the node's children are absorbed into.
        accumulator: C,
        /// Whether this node sandboxes its children's marks.
        marks: MarkScope,
    },
}

/// One post-order fold over an open node space.
///
/// Implementations are driven by [`open_fold_post_order`]; see the module
/// documentation for which hook owns which decision.
pub trait OpenFold {
    /// The node vocabulary — often a computed value, not a stored graph node.
    type Node;
    /// Cycle-guard key for the nodes that participate in marking.
    type Mark: Clone + Ord;
    /// The answer a subtree yields.
    type Value;
    /// The in-progress combination of one node's child answers.
    type Accumulator;

    /// Appends `node`'s children to `out`, in fold order.
    fn successors(&mut self, node: &Self::Node, out: &mut Vec<Self::Node>);

    /// The cycle-guard key for `node`, or `None` to fold it every time it
    /// is reached.
    fn mark(&mut self, node: &Self::Node) -> Option<Self::Mark>;

    /// Answers for `node` directly or opens its accumulator.
    fn enter(&mut self, node: &Self::Node) -> FoldEnter<Self::Value, Self::Accumulator>;

    /// Merges one child's answer into the accumulator.
    ///
    /// `value` is `None` for a child whose subtree yielded nothing — an
    /// empty answer, a refused (marked or too-deep) child, or a child cut
    /// off by an earlier break. Returning `ControlFlow::Break` skips the
    /// node's remaining children and finishes it now.
    fn absorb(
        &mut self,
        accumulator: &mut Self::Accumulator,
        child: &Self::Node,
        value: Option<Self::Value>,
    ) -> ControlFlow<()>;

    /// Turns the finished accumulator into the node's own answer.
    ///
    /// This runs on the unwind, so it is the place to rewrite a subtree's
    /// answer before the parent absorbs it.
    fn finish(&mut self, node: &Self::Node, accumulator: Self::Accumulator) -> Option<Self::Value>;
}

/// The live mark set, with the order its marks were set in.
///
/// Isolation is an undo, not a copy: a node that sandboxes its children
/// records how many marks were set when it was entered, and taking back
/// everything its subtree added is dropping the marks past that length. A
/// mark is only ever set when absent, so the log and the set always hold the
/// same keys.
struct UndoMarks<K> {
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

    fn contains(&self, key: &K) -> bool {
        self.set.contains(key)
    }

    /// The number of marks set, which is also the position of the next one.
    fn len(&self) -> usize {
        self.log.len()
    }

    /// Sets a mark the caller has checked is absent.
    fn insert(&mut self, key: K) {
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

struct Frame<F: OpenFold> {
    node: F::Node,
    accumulator: F::Accumulator,
    /// Where this node's children start on the walk's shared child stack.
    /// They are pushed in reverse, so the next one to fold is on top, and the
    /// frame has none left once the stack is back down to `base`.
    base: usize,
    sandbox: Option<Sandbox>,
    broke: bool,
}

/// The state of one walk: its frames, their children, and the live marks.
struct Walk<F: OpenFold> {
    stack: Vec<Frame<F>>,
    /// Every open frame's unfolded children, the deepest frame's on top.
    pending: Vec<F::Node>,
    /// The buffer `successors` fills, reused for every node.
    scratch: Vec<F::Node>,
    marks: UndoMarks<F::Mark>,
}

impl<F: OpenFold> Walk<F> {
    /// Opens a frame for `node`, whose own key (if it has one) is already the
    /// last mark set; `entry_marks` is how many marks were set before it.
    fn push_frame(
        &mut self,
        fold: &mut F,
        node: F::Node,
        accumulator: F::Accumulator,
        scope: MarkScope,
        entry_marks: usize,
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
        });
    }
}

/// Folds the space reachable from `root` in post-order under `fold`'s rules.
///
/// Returns the root's answer. See [`OpenFold`] and the module documentation
/// for the contract; the walk itself is iterative, so recursion depth never
/// limits the foldable space.
///
/// ```rust
/// use core::ops::ControlFlow;
/// use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, open_fold_post_order};
///
/// // C++-style member lookup: a class answers with its own declaration or
/// // with the answer of exactly one yielding base; two yielding bases are
/// // an ambiguity. Per-path marks fold a diamond's shared base per route.
/// struct Lookup<'a> {
///     bases: &'a [&'a [usize]],
///     owns: &'a [Option<&'static str>],
/// }
///
/// impl OpenFold for Lookup<'_> {
///     type Node = usize;
///     type Mark = usize;
///     type Value = &'static str;
///     type Accumulator = Result<Option<&'static str>, ()>;
///
///     fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
///         out.extend_from_slice(self.bases[*node]);
///     }
///     fn mark(&mut self, node: &usize) -> Option<usize> {
///         Some(*node)
///     }
///     fn enter(&mut self, node: &usize) -> FoldEnter<&'static str, Self::Accumulator> {
///         match self.owns[*node] {
///             Some(name) => FoldEnter::Leaf(Some(name)),
///             None => FoldEnter::Fold { accumulator: Ok(None), marks: MarkScope::Isolated },
///         }
///     }
///     fn absorb(
///         &mut self,
///         accumulator: &mut Self::Accumulator,
///         _child: &usize,
///         value: Option<&'static str>,
///     ) -> ControlFlow<()> {
///         if let Some(value) = value {
///             *accumulator = match accumulator {
///                 Ok(None) => Ok(Some(value)),
///                 _ => Err(()), // a second yielding base: ambiguous
///             };
///         }
///         ControlFlow::Continue(())
///     }
///     fn finish(&mut self, _node: &usize, accumulator: Self::Accumulator) -> Option<&'static str> {
///         accumulator.unwrap_or_default()
///     }
/// }
///
/// // 0 derives 1 and 2; both derive 3, which declares the member: the
/// // diamond folds 3 once per route, so the name is ambiguous at 0.
/// let bases: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
/// let owns = &[None, None, None, Some("member")];
/// let mut lookup = Lookup { bases, owns };
/// assert_eq!(
///     open_fold_post_order(&mut lookup, 0, OpenFoldConfig::new()),
///     None,
/// );
/// // 1 alone sees it unambiguously.
/// assert_eq!(
///     open_fold_post_order(&mut lookup, 1, OpenFoldConfig::new()),
///     Some("member"),
/// );
/// ```
pub fn open_fold_post_order<F: OpenFold>(
    fold: &mut F,
    root: F::Node,
    config: OpenFoldConfig,
) -> Option<F::Value> {
    let mut walk = Walk::<F> {
        stack: Vec::new(),
        pending: Vec::new(),
        scratch: Vec::new(),
        marks: UndoMarks::new(),
    };

    let root_key = fold.mark(&root);
    if let Some(key) = root_key {
        walk.marks.insert(key);
    }
    match fold.enter(&root) {
        FoldEnter::Leaf(value) => return value,
        FoldEnter::Fold {
            accumulator,
            marks: scope,
        } => walk.push_frame(fold, root, accumulator, scope, 0),
    }

    let mut result = None;
    while let Some(frame) = walk.stack.last_mut() {
        let child = if frame.broke || walk.pending.len() <= frame.base {
            None
        } else {
            walk.pending.pop()
        };
        let Some(child) = child else {
            let Some(frame) = walk.stack.pop() else { break };
            // A node stopped early leaves its remaining children unfolded.
            walk.pending.truncate(frame.base);
            let value = fold.finish(&frame.node, frame.accumulator);
            if let Some(sandbox) = frame.sandbox {
                walk.marks.truncate(sandbox.entry);
            }
            match walk.stack.last_mut() {
                Some(parent) => {
                    if fold
                        .absorb(&mut parent.accumulator, &frame.node, value)
                        .is_break()
                    {
                        parent.broke = true;
                    }
                }
                None => result = value,
            }
            continue;
        };

        // An isolated parent resets its children's shared start state: marks
        // as seen at its own entry, plus its own key. For its first child
        // that is the state it already has.
        if let Some(sandbox) = &frame.sandbox {
            walk.marks.truncate(sandbox.with_key);
        }

        let refused_by_depth = config
            .max_depth
            .is_some_and(|max_depth| walk.stack.len() > max_depth);
        let child_key = fold.mark(&child);
        let refused = refused_by_depth
            || child_key
                .as_ref()
                .is_some_and(|key| walk.marks.contains(key));
        if refused {
            if let Some(parent) = walk.stack.last_mut()
                && fold
                    .absorb(&mut parent.accumulator, &child, None)
                    .is_break()
            {
                parent.broke = true;
            }
            continue;
        }

        let entry_marks = walk.marks.len();
        if let Some(key) = child_key {
            walk.marks.insert(key);
        }
        match fold.enter(&child) {
            FoldEnter::Leaf(value) => {
                if let Some(parent) = walk.stack.last_mut()
                    && fold
                        .absorb(&mut parent.accumulator, &child, value)
                        .is_break()
                {
                    parent.broke = true;
                }
            }
            FoldEnter::Fold {
                accumulator,
                marks: scope,
            } => walk.push_frame(fold, child, accumulator, scope, entry_marks),
        }
    }
    result
}

#[cfg(test)]
mod tests;
