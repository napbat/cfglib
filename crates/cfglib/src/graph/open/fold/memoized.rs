//! The post-order fold that reuses a finished node's answer.

extern crate alloc;

use alloc::collections::BTreeMap;

use super::{FoldEnter, OpenFold, OpenFoldConfig, Walk};

/// What the memoized walk keeps for one open frame.
struct Memo<K> {
    /// The key the node's answer is cached under.
    key: Option<K>,
    /// The node's mark, which names the frame while it is open.
    mark: Option<K>,
    /// The stack position of the shallowest open frame that a refusal in
    /// this frame's subtree named, or `usize::MAX` for none.
    ///
    /// The frame's answer depends on where the walk entered a cycle exactly
    /// when this position lies below the frame's own: the refused node is an
    /// ancestor whose answer was still missing. A refusal of the frame itself
    /// closes a cycle that the frame contains whole, so it keeps the answer
    /// cacheable.
    reaches: usize,
}

/// The stack positions of the open frames, by memo key and by mark.
///
/// No two open frames share a key or a mark: a child whose key or mark
/// belongs to an open frame is refused before it opens a frame.
struct OpenFrames<K> {
    by_key: BTreeMap<K, usize>,
    by_mark: BTreeMap<K, usize>,
}

impl<K: Clone + Ord> OpenFrames<K> {
    fn open(&mut self, memo: &Memo<K>, position: usize) {
        if let Some(key) = &memo.key {
            self.by_key.insert(key.clone(), position);
        }
        if let Some(mark) = &memo.mark {
            self.by_mark.insert(mark.clone(), position);
        }
    }

    fn close(&mut self, memo: &Memo<K>) {
        if let Some(key) = &memo.key {
            self.by_key.remove(key);
        }
        if let Some(mark) = &memo.mark {
            self.by_mark.remove(mark);
        }
    }
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

/// Folds the space reachable from `root` in post-order, folding each node
/// that has an [`OpenFold::memo_key`] once.
///
/// The walk is the one [`open_fold_post_order`](super::open_fold_post_order)
/// makes, with one change: a node with a memo key that has finished stores
/// its answer under that key, and every later reach of that key absorbs a
/// clone of the answer without calling [`enter`](OpenFold::enter) or
/// [`successors`](OpenFold::successors). A fold whose nodes have no memo key
/// walks exactly as the plain driver does.
///
/// - **The contract.** The cached answer is reused on every route, so it
///   must not depend on the route. A fold that needs a fact of the route
///   carries it in its value and decides in [`absorb`](OpenFold::absorb). On
///   a space without cycles, a fold that marks each node
///   [`MarkScope::Isolated`](super::MarkScope::Isolated) then returns the
///   answer of the plain driver while it folds each key once instead of once
///   per route.
/// - **Cycles.** A child whose memo key or mark belongs to a node still open
///   on the current route is refused and absorbed as `None`, as the plain
///   driver refuses a marked child. The answers of the nodes between the
///   refusing node and the refused one depend on where the walk entered the
///   cycle, so the driver does not cache them; the refused node's own answer,
///   and the answers above it, stay cacheable.
/// - **Other refusals.** A child that a persisting mark of a finished node
///   refuses ([`MarkScope::Shared`](super::MarkScope::Shared)) contributes
///   `None` and does not stop the caching: the fold chose to prune that
///   reach. The depth bound refuses a child below it as the plain driver
///   does, but it bounds the walk, not each route: an answer cached at one
///   depth is reused at every other.
///
/// Each node with a memo key costs one fold, plus one table lookup of
/// `O(log n)` per reach. Results are deterministic, because the successor
/// order is the plain driver's.
///
/// ```rust
/// use core::ops::ControlFlow;
/// use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, open_fold_post_order_memoized};
///
/// /// The subobject that declares a member: its class, and whether its
/// /// route passed a virtual base, which makes it one shared subobject.
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// struct Subobject {
///     class: usize,
///     shared: bool,
/// }
///
/// /// The class being folded, and what its bases yielded so far.
/// struct Found {
///     class: usize,
///     subobject: Option<Subobject>,
///     ambiguous: bool,
/// }
///
/// // A simplified C++ member lookup. Each base carries a flag that says
/// // whether it is virtual. A class's answer is the same on every route, so
/// // the memoized driver folds each class once.
/// struct Lookup<'a> {
///     bases: &'a [&'a [(usize, bool)]],
///     owns: &'a [bool],
/// }
///
/// impl OpenFold for Lookup<'_> {
///     type Node = usize;
///     type Mark = usize;
///     type Value = Subobject;
///     type Accumulator = Found;
///
///     fn successors(&mut self, node: &usize, out: &mut Vec<usize>) {
///         out.extend(self.bases[*node].iter().map(|&(base, _)| base));
///     }
///     fn mark(&mut self, node: &usize) -> Option<usize> {
///         Some(*node)
///     }
///     fn memo_key(&mut self, node: &usize) -> Option<usize> {
///         Some(*node)
///     }
///     fn enter(&mut self, node: &usize) -> FoldEnter<Subobject, Found> {
///         if self.owns[*node] {
///             FoldEnter::Leaf(Some(Subobject { class: *node, shared: false }))
///         } else {
///             let accumulator = Found { class: *node, subobject: None, ambiguous: false };
///             FoldEnter::Fold { accumulator, marks: MarkScope::Isolated }
///         }
///     }
///     fn absorb(
///         &mut self,
///         found: &mut Found,
///         base: &usize,
///         value: Option<Subobject>,
///     ) -> ControlFlow<()> {
///         if let Some(mut value) = value {
///             // The route decides in `absorb`: through a virtual base, the
///             // declaring subobject is the shared one.
///             value.shared |= self.bases[found.class].contains(&(*base, true));
///             match found.subobject {
///                 None => found.subobject = Some(value),
///                 // Two routes to one shared subobject find one member.
///                 Some(seen) if seen == value && seen.shared => {}
///                 Some(_) => found.ambiguous = true,
///             }
///         }
///         ControlFlow::Continue(())
///     }
///     fn finish(&mut self, _node: &usize, found: Found) -> Option<Subobject> {
///         if found.ambiguous { None } else { found.subobject }
///     }
/// }
///
/// // 0 derives 1 and 2; both derive 3, which declares the member. The walk
/// // folds 3 once and hands its answer to both routes.
/// let owns = &[false, false, false, true];
/// let plain: &[&[(usize, bool)]] = &[&[(1, false), (2, false)], &[(3, false)], &[(3, false)], &[]];
/// let mut lookup = Lookup { bases: plain, owns };
/// // Two distinct subobjects of 3 declare the member: ambiguous.
/// assert_eq!(
///     open_fold_post_order_memoized(&mut lookup, 0, OpenFoldConfig::new()),
///     None,
/// );
/// // Virtual inheritance of 3 leaves one shared subobject.
/// let virtual_bases: &[&[(usize, bool)]] = &[&[(1, false), (2, false)], &[(3, true)], &[(3, true)], &[]];
/// let mut lookup = Lookup { bases: virtual_bases, owns };
/// assert_eq!(
///     open_fold_post_order_memoized(&mut lookup, 0, OpenFoldConfig::new()),
///     Some(Subobject { class: 3, shared: true }),
/// );
/// ```
pub fn open_fold_post_order_memoized<F: OpenFold>(
    fold: &mut F,
    root: F::Node,
    config: OpenFoldConfig,
) -> Option<F::Value>
where
    F::Value: Clone,
{
    let mut walk = Walk::<F, Memo<F::Mark>>::new();
    let mut open = OpenFrames {
        by_key: BTreeMap::new(),
        by_mark: BTreeMap::new(),
    };
    let mut answers: BTreeMap<F::Mark, Option<F::Value>> = BTreeMap::new();

    let memo = Memo {
        key: fold.memo_key(&root),
        mark: fold.mark(&root),
        reaches: usize::MAX,
    };
    if let Some(mark) = &memo.mark {
        walk.marks.insert(mark.clone());
    }
    match fold.enter(&root) {
        FoldEnter::Leaf(value) => return value,
        FoldEnter::Fold {
            accumulator,
            marks: scope,
        } => {
            open.open(&memo, 0);
            walk.push_frame(fold, root, accumulator, scope, 0, memo);
        }
    }

    let mut result = None;
    while !walk.stack.is_empty() {
        let Some(child) = walk.next_child() else {
            let Some((node, value, memo)) = walk.finish_top(fold) else {
                break;
            };
            open.close(&memo);
            let position = walk.stack.len();
            let Some(parent) = walk.stack.last_mut() else {
                result = value;
                break;
            };
            parent.extra.reaches = parent.extra.reaches.min(memo.reaches);
            if memo.reaches >= position
                && let Some(key) = memo.key
            {
                answers.insert(key, value.clone());
            }
            walk.absorb(fold, &node, value);
            continue;
        };

        if !walk.admit_child(config.max_depth) {
            walk.absorb(fold, &child, None);
            continue;
        }
        let key = fold.memo_key(&child);
        if let Some(key) = &key {
            if let Some(answer) = answers.get(key) {
                let answer = answer.clone();
                walk.absorb(fold, &child, answer);
                continue;
            }
            if let Some(&target) = open.by_key.get(key) {
                refuse_open(&mut walk, fold, &child, target);
                continue;
            }
        }
        let mark = fold.mark(&child);
        if let Some(mark) = &mark
            && walk.marks.contains(mark)
        {
            match open.by_mark.get(mark) {
                Some(&target) => refuse_open(&mut walk, fold, &child, target),
                None => walk.absorb(fold, &child, None),
            }
            continue;
        }

        let entry_marks = walk.marks.len();
        if let Some(mark) = &mark {
            walk.marks.insert(mark.clone());
        }
        match fold.enter(&child) {
            FoldEnter::Leaf(value) => {
                if let Some(key) = key {
                    answers.insert(key, value.clone());
                }
                walk.absorb(fold, &child, value);
            }
            FoldEnter::Fold {
                accumulator,
                marks: scope,
            } => {
                let memo = Memo {
                    key,
                    mark,
                    reaches: usize::MAX,
                };
                open.open(&memo, walk.stack.len());
                walk.push_frame(fold, child, accumulator, scope, entry_marks, memo);
            }
        }
    }
    result
}
