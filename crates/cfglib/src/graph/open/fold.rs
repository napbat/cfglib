//! Post-order folding over an open node space.
//!
//! A search can only stop; a fold computes each node's answer from its
//! children's answers on the unwind. [`open_fold`] owns the frame
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
//!
//! The [`Revisit`] policy of the [`OpenFoldConfig`] decides what the fold
//! does with a marked node it reaches again:
//!
//! - [`Revisit::Refuse`] absorbs it as `None` while its mark is live.
//!   `Isolated` then folds a shared descendant once per route, which is
//!   exponential in a stack of diamonds, and `Shared` refuses the second
//!   route, which hides it from a fold that must see both routes.
//! - [`Revisit::Memoize`] folds each marked node once and hands its finished
//!   answer to every later route, so the fold still sees each route but pays
//!   one fold per mark. That answer must not depend on the route: a fold
//!   that needs a fact of the route carries it in its value and decides in
//!   [`OpenFold::absorb`].
//! - [`Revisit::Iterate`] is the policy for a space whose cycles carry
//!   answers around: the names a module exports through `export *` cycles,
//!   or the members visible through cyclic includes. The other policies
//!   refuse a node that is still open, which makes each answer on a cycle
//!   depend on where the walk entered it and loses what a value brings back
//!   around the cycle. `Iterate` keeps one answer per mark, finds the cycles
//!   of marked nodes with Tarjan's lowlinks as it walks, and re-folds each
//!   cycle until a round changes nothing, so monotone hooks reach the least
//!   fixpoint from any root.
//!
//! [`open_fold_in`] takes the [`OpenFoldMemo`] of finished answers from the
//! caller, so a pass that folds from many roots over one space folds each
//! marked node once in total.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;
use core::ops::ControlFlow;

mod iterate;
mod memoize;
mod refuse;
mod walk;

/// Bounds and revisit policy of one open fold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenFoldConfig {
    /// Refuses children more than this many levels below the root, or `None`
    /// for no bound.
    ///
    /// The root is depth zero, and a re-fold of a cycle member under
    /// [`Revisit::Iterate`] runs at the depth of the cycle's first node. The
    /// bound is silent by design: a refused child is absorbed as `None`, the
    /// contribution a marked child makes, and no error reports the cut.
    ///
    /// Under [`Revisit::Memoize`] and [`Revisit::Iterate`], the answer of a
    /// node whose subtree the bound cut depends on the depth the walk
    /// reached the node at, as an answer on a cycle depends on where the walk
    /// entered it, so neither policy stores it: a later reach folds the node
    /// again. A stored answer is the node's whole answer, and a later reach
    /// reuses it at any depth, also where the bound would cut a fresh fold:
    /// the bound limits the walk that computes answers, not the routes a
    /// stored answer covers.
    pub max_depth: Option<usize>,
    /// What the fold does with a marked node it reaches again.
    pub revisit: Revisit,
}

/// What an open fold does with a marked node it reaches again.
///
/// Every policy folds a node without a [`mark`](OpenFold::mark) on each
/// reach, and every policy refuses a marked node that is still open on the
/// current route, except that [`Revisit::Iterate`] absorbs its provisional
/// answer instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Revisit {
    /// Absorbs a reached node whose mark is live as `None`.
    ///
    /// The [`MarkScope`] of each folded node decides how long the marks of
    /// its subtree live, so a diamond's shared base folds once per route
    /// (`Isolated`) or once per walk (`Shared`). The policy ignores the
    /// [`OpenFoldMemo`] of [`open_fold_in`].
    #[default]
    Refuse,
    /// Folds each marked node once and hands a clone of its finished answer
    /// to every later reach of its mark, without calling
    /// [`enter`](OpenFold::enter) or [`successors`](OpenFold::successors)
    /// again.
    ///
    /// - **The contract.** The stored answer is reused on every route, so it
    ///   must not depend on the route. A fold that needs a fact of the route
    ///   carries it in its value and decides in [`absorb`](OpenFold::absorb).
    ///   On a space without cycles, a fold that marks each node
    ///   [`MarkScope::Isolated`] then returns the answer of
    ///   [`Revisit::Refuse`] while it folds each mark once instead of once per
    ///   route.
    /// - **Cycles.** A child whose mark belongs to a node still open on the
    ///   current route is refused and absorbed as `None`. The answers of the
    ///   nodes between the refusing node and the refused one depend on where
    ///   the walk entered the cycle, so they are not stored; the refused
    ///   node's own answer, and the answers above it, are.
    /// - **Route-dependent nodes.** A node for which
    ///   [`memoize`](OpenFold::memoize) returns `false` keeps its answer to
    ///   its route: its mark still guards the route, so a cycle back to it is
    ///   refused, but its answer is not stored and every reach folds it
    ///   again.
    /// - **Depth.** A node whose subtree the depth bound cut is not stored
    ///   either (see [`OpenFoldConfig::max_depth`]).
    /// - **Marks.** A stored answer is handed over whatever the live marks
    ///   are, so the [`MarkScope`] matters only for a node whose answer was
    ///   not stored: a reach of it that a persisting mark refuses
    ///   ([`MarkScope::Shared`]) contributes `None`, and other reaches fold it
    ///   again.
    /// - **The table.** The policy reads the [`OpenFoldMemo`] of
    ///   [`open_fold_in`] and stores every answer it may reuse there, the
    ///   root's included.
    ///
    /// Each marked node costs one fold, plus one table lookup of `O(log n)`
    /// per reach. Results are deterministic, because the successor order is
    /// fixed.
    Memoize,
    /// Keeps one answer per mark and re-folds every cycle of marked nodes
    /// until its answers are stable.
    ///
    /// The other policies refuse a node that is still open on the route, so
    /// a fold over a cyclic space gets an answer that misses what travels
    /// around the cycle and that depends on where the walk entered it. This
    /// policy finds the strongly connected components of marked nodes with
    /// Tarjan's lowlinks as it walks, and re-folds the nodes of each cycle
    /// until a round changes no answer. Then every node of the cycle has its
    /// least fixpoint answer, whichever node the walk entered first.
    ///
    /// - **The walk.** A marked node that is still open absorbs its current,
    ///   provisional answer, which starts as `None`; a finished marked node
    ///   absorbs its final answer. Every marked node is memoized, so the
    ///   policy ignores the [`MarkScope`] of [`FoldEnter::Fold`].
    /// - **Iteration.** When the first node of a cycle finishes, each member
    ///   of the cycle is re-folded in the order its first fold finished: the
    ///   walk calls [`enter`](OpenFold::enter), absorbs each successor's
    ///   current answer without descending into members, and calls
    ///   [`finish`](OpenFold::finish). Rounds repeat until no answer changes,
    ///   and the answers become final. Hooks with side effects observe a
    ///   member's hooks once per round.
    /// - **The contract.** `enter`, `absorb` and `finish` must be
    ///   deterministic, and monotone in the absorbed answers over a domain of
    ///   finite height; or `max_rounds` bounds the rounds. A node on a cycle
    ///   needs a mark: a cycle that passes a marked node and a node without a
    ///   mark fails with [`OpenFoldError::UnmarkedCycle`]. A cycle of nodes
    ///   without marks is never closed, so only the depth bound stops it.
    /// - **The table.** The policy stores each final answer in the
    ///   [`OpenFoldMemo`] of [`open_fold_in`] and reuses the final answers it
    ///   finds there. Provisional answers never reach the table, so after an
    ///   error it holds only the answers that became final before it. Nor
    ///   does an answer that the depth bound cut, for a cycle the cut of any
    ///   fold of a member (see [`OpenFoldConfig::max_depth`]). The policy
    ///   stores every other marked node's final answer and does not read
    ///   [`memoize`](OpenFold::memoize).
    ///
    /// On a space without cycles this walks as [`Revisit::Memoize`] does:
    /// `O(n + m)` hook calls. Each cycle adds `O(rounds × (n_scc + m_scc))`
    /// hook calls. The successor order is fixed, so results are deterministic.
    Iterate {
        /// The re-fold rounds one cycle may take, or `None` for no limit. A
        /// cycle whose answers still change after them fails with
        /// [`OpenFoldError::RoundLimitExceeded`].
        max_rounds: Option<usize>,
    },
}

/// The finished answers of open folds by mark, owned by the caller of
/// [`open_fold_in`] so that they persist across calls.
///
/// [`Revisit::Memoize`] reads and writes the table, [`Revisit::Iterate`]
/// stores its final answers in it and reuses the final answers it finds
/// there, and [`Revisit::Refuse`] ignores it. An answer is stored under the
/// [`mark`](OpenFold::mark) of its node, and `None` is a stored answer too:
/// the fold of the node yielded nothing.
///
/// A stored answer belongs to the fold that computed it. Reuse a table only
/// across calls with the same fold (the same hooks over the same space) and
/// the same policy, since `Memoize` and `Iterate` answer a cycle
/// differently. Neither policy stores an answer that the depth bound cut, so
/// a stored answer does not depend on the
/// [`max_depth`](OpenFoldConfig::max_depth) of the call that computed it,
/// and calls with different bounds may share a table. When the fold or the
/// policy changes, [`clear`](Self::clear) the table or
/// [`remove`](Self::remove) the answers that went stale.
#[derive(Debug, Clone)]
pub struct OpenFoldMemo<K, V> {
    answers: BTreeMap<K, Stored<V>>,
}

/// One entry of an [`OpenFoldMemo`].
///
/// The open entries keep `Revisit::Iterate` to one table lookup per reach.
/// With the provisional answers in a second map beside the final ones, every
/// first reach of a node missed in both and every finished node moved from
/// one map to the other: `open_fold_fixpoint_diamonds` ran 1.53x and
/// `open_fold_fixpoint_export_ring` 1.35x the time of one table.
#[derive(Debug, Clone)]
enum Stored<V> {
    /// A final answer.
    Final(Option<V>),
    /// A marked node whose cycle `Revisit::Iterate` is still folding: its
    /// position on that walk's member stack, which holds its provisional
    /// answer. The walk replaces the entry once the cycle is final and
    /// removes it if it ends without an answer, so no caller sees one.
    Open(usize),
}

impl<K, V> Default for OpenFoldMemo<K, V> {
    fn default() -> Self {
        Self {
            answers: BTreeMap::new(),
        }
    }
}

impl<K: Ord, V> OpenFoldMemo<K, V> {
    /// The answer stored for `mark`: `None` when there is none, and
    /// `Some(None)` when the fold of its node yielded nothing.
    #[must_use]
    pub fn get(&self, mark: &K) -> Option<Option<&V>> {
        match self.answers.get(mark)? {
            Stored::Final(value) => Some(value.as_ref()),
            Stored::Open(_) => None,
        }
    }

    /// Removes the answer stored for `mark` and returns it, so the next fold
    /// that reaches the mark folds its node again.
    ///
    /// The answers of the nodes that absorbed it stay stored; remove them
    /// too when the removed answer went stale.
    pub fn remove(&mut self, mark: &K) -> Option<Option<V>> {
        match self.answers.remove(mark)? {
            Stored::Final(value) => Some(value),
            Stored::Open(_) => None,
        }
    }

    /// Removes every stored answer.
    pub fn clear(&mut self) {
        self.answers.clear();
    }

    /// The number of stored answers.
    #[must_use]
    pub fn answer_count(&self) -> usize {
        self.answers.len()
    }

    /// Whether the table stores no answer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.answers.is_empty()
    }
}

/// Why [`open_fold`] or [`open_fold_in`] stopped without an answer.
///
/// Only [`Revisit::Iterate`] fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenFoldError {
    /// The answers of a cycle still changed in the last round that the
    /// `max_rounds` of [`Revisit::Iterate`] allows.
    RoundLimitExceeded {
        /// The configured round limit.
        limit: usize,
        /// The re-fold rounds that ran.
        rounds: usize,
    },
    /// A node without a mark lies on a cycle with a marked node, so the walk
    /// has no key to hold its provisional answer.
    UnmarkedCycle,
}

impl fmt::Display for OpenFoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::RoundLimitExceeded { limit, rounds } => write!(
                formatter,
                "open fold round limit {limit} exceeded: a cycle still changed after {rounds} rounds"
            ),
            Self::UnmarkedCycle => formatter.write_str("a node without a mark lies on a cycle"),
        }
    }
}

impl core::error::Error for OpenFoldError {}

/// How far the marks written inside one folded node's subtree reach.
///
/// [`Revisit::Iterate`] ignores the scope.
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
    /// The node is still marked (per [`OpenFold::mark`]): under
    /// [`Revisit::Refuse`] a later reach under a persisting scope is refused
    /// rather than re-answered, and the other policies store the answer.
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
/// Implementations are driven by [`open_fold`] or [`open_fold_in`]; see the
/// module documentation for which hook owns which decision.
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
    ///
    /// [`Revisit::Memoize`] and [`Revisit::Iterate`] also store the answer of
    /// `node` under this key, so nodes that share a key share one answer.
    fn mark(&mut self, node: &Self::Node) -> Option<Self::Mark>;

    /// Whether [`Revisit::Memoize`] may store the answer of `node`, a marked
    /// node, and hand it to later reaches of its mark.
    ///
    /// Return `false` for a node whose answer depends on its route beyond
    /// what its value carries, such as a node on a cycle whose answer
    /// depends on where the walk entered the cycle. The node's mark still
    /// guards the current route, so a cycle back to it is refused, but every
    /// reach folds it again. The walk asks once the node's answer is ready,
    /// so the hook may depend on what the fold observed while folding it.
    /// The default stores every marked node's answer. [`Revisit::Refuse`]
    /// stores nothing, and [`Revisit::Iterate`] does not read this hook.
    fn memoize(&mut self, _node: &Self::Node) -> bool {
        true
    }

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

/// Folds the space reachable from `root` in post-order under `fold`'s rules
/// and the policy of `config`, and returns the root's answer.
///
/// This is [`open_fold_in`] with a fresh [`OpenFoldMemo`], so the answers that
/// [`Revisit::Memoize`] and [`Revisit::Iterate`] store last for this call.
/// See [`OpenFold`], [`Revisit`] and the module documentation for the
/// contract; the walk itself is iterative, so recursion depth never limits
/// the foldable space.
///
/// # Errors
///
/// Only [`Revisit::Iterate`] fails: with
/// [`OpenFoldError::RoundLimitExceeded`] when a cycle still changes after
/// its `max_rounds`, and with [`OpenFoldError::UnmarkedCycle`] when a node
/// without a mark lies on a cycle with a marked node.
///
/// # Examples
///
/// Under [`Revisit::Refuse`], per-path marks fold a diamond's shared base
/// once per route:
///
/// ```rust
/// use core::ops::ControlFlow;
/// use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, open_fold};
///
/// // C++-style member lookup: a class answers with its own declaration or
/// // with the answer of exactly one yielding base; two yielding bases are
/// // an ambiguity.
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
/// assert_eq!(open_fold(&mut lookup, 0, OpenFoldConfig::default()), Ok(None));
/// // 1 alone sees it unambiguously.
/// assert_eq!(open_fold(&mut lookup, 1, OpenFoldConfig::default()), Ok(Some("member")));
/// ```
///
/// [`Revisit::Memoize`] folds a class once and hands its answer to every
/// route, so a fact of the route travels in the value:
///
/// ```rust
/// use core::ops::ControlFlow;
/// use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, Revisit, open_fold};
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
/// // whether it is virtual. A class's answer is the same on every route.
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
/// let memoize = OpenFoldConfig { revisit: Revisit::Memoize, ..OpenFoldConfig::default() };
/// let owns = &[false, false, false, true];
/// let plain: &[&[(usize, bool)]] = &[&[(1, false), (2, false)], &[(3, false)], &[(3, false)], &[]];
/// let mut lookup = Lookup { bases: plain, owns };
/// // Two distinct subobjects of 3 declare the member: ambiguous.
/// assert_eq!(open_fold(&mut lookup, 0, memoize), Ok(None));
/// // Virtual inheritance of 3 leaves one shared subobject.
/// let virtual_bases: &[&[(usize, bool)]] = &[&[(1, false), (2, false)], &[(3, true)], &[(3, true)], &[]];
/// let mut lookup = Lookup { bases: virtual_bases, owns };
/// assert_eq!(open_fold(&mut lookup, 0, memoize), Ok(Some(Subobject { class: 3, shared: true })));
/// ```
///
/// [`Revisit::Iterate`] carries the answers around a cycle:
///
/// ```rust
/// use std::collections::BTreeSet;
/// use core::ops::ControlFlow;
/// use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, Revisit, open_fold};
///
/// /// How a module re-exports another.
/// enum Export {
///     /// `export * from "..."`
///     All,
///     /// `export { from as to } from "..."`
///     Renamed(&'static str, &'static str),
/// }
///
/// /// The names each module exports: its own, and what its re-exports bring.
/// struct Exports<'a> {
///     own: &'a [&'a [&'static str]],
///     reexports: &'a [&'a [(usize, Export)]],
/// }
///
/// impl OpenFold for Exports<'_> {
///     type Node = usize;
///     type Mark = usize;
///     type Value = BTreeSet<&'static str>;
///     type Accumulator = (usize, BTreeSet<&'static str>);
///
///     fn successors(&mut self, module: &usize, out: &mut Vec<usize>) {
///         out.extend(self.reexports[*module].iter().map(|(target, _)| *target));
///     }
///     fn mark(&mut self, module: &usize) -> Option<usize> {
///         Some(*module)
///     }
///     fn enter(&mut self, module: &usize) -> FoldEnter<Self::Value, Self::Accumulator> {
///         let names = self.own[*module].iter().copied().collect();
///         FoldEnter::Fold { accumulator: (*module, names), marks: MarkScope::Shared }
///     }
///     fn absorb(
///         &mut self,
///         (module, names): &mut Self::Accumulator,
///         target: &usize,
///         value: Option<Self::Value>,
///     ) -> ControlFlow<()> {
///         let exported = value.unwrap_or_default();
///         for (reexported, export) in self.reexports[*module] {
///             if reexported != target {
///                 continue;
///             }
///             match export {
///                 Export::All => names.extend(exported.iter().copied()),
///                 Export::Renamed(from, to) => {
///                     if exported.contains(from) {
///                         names.insert(*to);
///                     }
///                 }
///             }
///         }
///         ControlFlow::Continue(())
///     }
///     fn finish(&mut self, _module: &usize, (_, names): Self::Accumulator) -> Option<Self::Value> {
///         Some(names)
///     }
/// }
///
/// // Module 0 exports `x` and re-exports 1's `x` as `z`; module 1 re-exports
/// // all of 0. Both export `x` and `z`.
/// let own: &[&[&str]] = &[&["x"], &[]];
/// let reexports: &[&[(usize, Export)]] =
///     &[&[(1, Export::Renamed("x", "z"))], &[(0, Export::All)]];
/// let iterate = OpenFoldConfig {
///     revisit: Revisit::Iterate { max_rounds: None },
///     ..OpenFoldConfig::default()
/// };
/// let both = BTreeSet::from(["x", "z"]);
/// for module in [0, 1] {
///     let mut exports = Exports { own, reexports };
///     assert_eq!(open_fold(&mut exports, module, iterate), Ok(Some(both.clone())));
/// }
/// // Refusing 0 under 1 brings nothing back to 0.
/// let mut exports = Exports { own, reexports };
/// assert_eq!(
///     open_fold(&mut exports, 0, OpenFoldConfig::default()),
///     Ok(Some(BTreeSet::from(["x"]))),
/// );
/// ```
pub fn open_fold<F: OpenFold>(
    fold: &mut F,
    root: F::Node,
    config: OpenFoldConfig,
) -> Result<Option<F::Value>, OpenFoldError>
where
    F::Value: Clone + PartialEq,
{
    open_fold_in(&mut OpenFoldMemo::default(), fold, root, config)
}

/// Folds the space reachable from `root` as [`open_fold`] does, with the
/// table of finished answers owned by the caller.
///
/// [`Revisit::Memoize`] reads and writes `memo`, [`Revisit::Iterate`] stores
/// its final answers there and reuses the final answers it finds, and
/// [`Revisit::Refuse`] ignores it. A pass that folds from many roots over
/// one space keeps one table, so each marked node folds once over the whole
/// pass; [`OpenFoldMemo`] says when a table may be reused.
///
/// # Errors
///
/// Fails as [`open_fold`] does. The table then holds only answers that were
/// final before the failure.
///
/// # Examples
///
/// ```rust
/// use core::ops::ControlFlow;
/// use cfglib::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, OpenFoldMemo, Revisit, open_fold_in};
///
/// /// The longest chain of bases below a class, recording the classes folded.
/// struct Height<'a> {
///     bases: &'a [&'a [usize]],
///     folded: Vec<usize>,
/// }
///
/// impl OpenFold for Height<'_> {
///     type Node = usize;
///     type Mark = usize;
///     type Value = usize;
///     type Accumulator = usize;
///
///     fn successors(&mut self, class: &usize, out: &mut Vec<usize>) {
///         out.extend_from_slice(self.bases[*class]);
///     }
///     fn mark(&mut self, class: &usize) -> Option<usize> {
///         Some(*class)
///     }
///     fn enter(&mut self, class: &usize) -> FoldEnter<usize, usize> {
///         self.folded.push(*class);
///         FoldEnter::Fold { accumulator: 0, marks: MarkScope::Isolated }
///     }
///     fn absorb(&mut self, height: &mut usize, _base: &usize, below: Option<usize>) -> ControlFlow<()> {
///         *height = (*height).max(below.map_or(0, |below| below + 1));
///         ControlFlow::Continue(())
///     }
///     fn finish(&mut self, _class: &usize, height: usize) -> Option<usize> {
///         Some(height)
///     }
/// }
///
/// // 0 and 1 both derive 2, which derives 3.
/// let bases: &[&[usize]] = &[&[2], &[2], &[3], &[]];
/// let mut height = Height { bases, folded: Vec::new() };
/// let memoize = OpenFoldConfig { revisit: Revisit::Memoize, ..OpenFoldConfig::default() };
/// let mut memo = OpenFoldMemo::default();
/// assert_eq!(open_fold_in(&mut memo, &mut height, 0, memoize), Ok(Some(2)));
/// // The second root reuses the answer the first call stored for 2.
/// assert_eq!(open_fold_in(&mut memo, &mut height, 1, memoize), Ok(Some(2)));
/// assert_eq!(height.folded, [0, 2, 3, 1]);
/// assert_eq!(memo.get(&2), Some(Some(&1)));
/// assert_eq!(memo.answer_count(), 4);
/// ```
pub fn open_fold_in<F: OpenFold>(
    memo: &mut OpenFoldMemo<F::Mark, F::Value>,
    fold: &mut F,
    root: F::Node,
    config: OpenFoldConfig,
) -> Result<Option<F::Value>, OpenFoldError>
where
    F::Value: Clone + PartialEq,
{
    match config.revisit {
        Revisit::Refuse => Ok(refuse::run(fold, root, config.max_depth)),
        Revisit::Memoize => Ok(memoize::run(
            &mut memo.answers,
            fold,
            root,
            config.max_depth,
        )),
        Revisit::Iterate { max_rounds } => {
            iterate::run(&mut memo.answers, fold, root, config.max_depth, max_rounds)
        }
    }
}

#[cfg(test)]
mod tests;
