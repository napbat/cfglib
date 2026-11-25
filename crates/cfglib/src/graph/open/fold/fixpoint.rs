//! The post-order fold that iterates each cycle of marked nodes to a
//! fixpoint.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use super::{FoldEnter, MarkScope, OpenFold, Walk};

#[cfg(test)]
mod tests;

/// Bounds for one [`open_fold_fixpoint`] walk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenFixpointConfig {
    max_depth: Option<usize>,
    max_scc_rounds: Option<usize>,
}

impl OpenFixpointConfig {
    /// Creates the unbounded configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_depth: None,
            max_scc_rounds: None,
        }
    }

    /// Refuses children more than `depth` frames below the root.
    ///
    /// The root is depth zero, and a re-fold of a cycle member runs at the
    /// depth of the cycle's first node. A refused child is absorbed as
    /// `None`. The bound limits the walk, not each route: the answer of a node
    /// whose subtree the bound cut is memoized like any other.
    #[must_use]
    pub const fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Stops with [`OpenFixpointError::RoundLimit`] when a cycle's answers
    /// still change after `rounds` re-fold rounds.
    #[must_use]
    pub const fn with_max_scc_rounds(mut self, rounds: usize) -> Self {
        self.max_scc_rounds = Some(rounds);
        self
    }
}

/// Why [`open_fold_fixpoint`] stopped without an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenFixpointError {
    /// The answers of a cycle still changed in the last round that
    /// [`OpenFixpointConfig::with_max_scc_rounds`] allows.
    RoundLimit {
        /// The re-fold rounds that ran.
        rounds: usize,
    },
    /// A node without a mark lies on a cycle, so the walk has no key to hold
    /// its provisional answer.
    UnmarkedCycle,
}

impl fmt::Display for OpenFixpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundLimit { rounds } => {
                write!(
                    formatter,
                    "a cycle was still changing after {rounds} rounds"
                )
            }
            Self::UnmarkedCycle => formatter.write_str("a node without a mark lies on a cycle"),
        }
    }
}

impl core::error::Error for OpenFixpointError {}

/// What the frame of one node is folding.
#[derive(Clone, Copy)]
enum Role {
    /// The first fold of a node without a mark. `entry` is the index the
    /// next marked node would get when the frame opened.
    Unmarked { entry: usize },
    /// The first fold of a marked node, with its index in discovery order
    /// and its position on the member stack.
    Marked { index: usize, position: usize },
    /// A re-fold of the cycle member at this position of the member stack.
    Refold { position: usize },
}

/// What the fixpoint walk keeps for one frame.
#[derive(Clone, Copy)]
struct Fix {
    role: Role,
    /// The smallest index of an open member that this frame's subtree
    /// reached, or `usize::MAX`.
    lowlink: usize,
    /// Whether the node reached itself directly.
    self_loop: bool,
}

/// The answer the walk holds for one mark.
enum Slot<V> {
    /// A member of a cycle that is not finished, with its index and its
    /// current, provisional answer.
    Open { index: usize, value: Option<V> },
    /// A final answer.
    Done(Option<V>),
}

/// A marked node whose cycle is not finished.
struct Member<F: OpenFold> {
    mark: F::Mark,
    /// The node, while no frame folds it.
    node: Option<F::Node>,
    /// When its first fold finished; a round re-folds members in this order.
    finished: usize,
}

/// One cycle being iterated: the members from `base` up on the member stack.
struct Component {
    base: usize,
    /// The index of the member at `base`, the first one the walk reached.
    index: usize,
    /// The smallest index of an open node that a re-fold reached. A re-fold
    /// can reach a successor that an earlier fold stopped before, so the
    /// cycle can turn out to belong to a larger one that is still open.
    lowlink: usize,
    /// The member positions of the current round, in re-fold order.
    order: Vec<usize>,
    next: usize,
    rounds: usize,
    changed: bool,
}

/// What reaching a node gave.
enum Reached<N, V> {
    /// An answer for the node, without a frame.
    Answer(N, Option<V>),
    /// A frame for the node is open.
    Opened,
}

/// What finishing a frame gave.
enum Step<V> {
    /// The walk goes on.
    Continue,
    /// The root's final answer.
    Root(Option<V>),
}

struct Run<F: OpenFold> {
    walk: Walk<F, Fix>,
    slots: BTreeMap<F::Mark, Slot<F::Value>>,
    members: Vec<Member<F>>,
    components: Vec<Component>,
    next_index: usize,
    finished: usize,
    config: OpenFixpointConfig,
}

impl<F: OpenFold> Run<F>
where
    F::Value: Clone + PartialEq,
{
    fn push_frame(&mut self, fold: &mut F, node: F::Node, accumulator: F::Accumulator, fix: Fix) {
        // Every marked node is memoized, so no mark scope applies.
        self.walk
            .push_frame(fold, node, accumulator, MarkScope::Shared, 0, fix);
    }

    /// Answers `node` from its slot or as a leaf, or opens a frame for it.
    fn reach(&mut self, fold: &mut F, node: F::Node) -> Reached<F::Node, F::Value> {
        let Some(mark) = fold.mark(&node) else {
            return match fold.enter(&node) {
                FoldEnter::Leaf(value) => Reached::Answer(node, value),
                FoldEnter::Fold { accumulator, .. } => {
                    let fix = Fix {
                        role: Role::Unmarked {
                            entry: self.next_index,
                        },
                        lowlink: usize::MAX,
                        self_loop: false,
                    };
                    self.push_frame(fold, node, accumulator, fix);
                    Reached::Opened
                }
            };
        };
        match self.slots.get(&mark) {
            Some(Slot::Done(value)) => return Reached::Answer(node, value.clone()),
            Some(Slot::Open { index, value }) => {
                let index = *index;
                let value = value.clone();
                if let Some(top) = self.walk.stack.last_mut() {
                    top.extra.lowlink = top.extra.lowlink.min(index);
                    if matches!(top.extra.role, Role::Marked { index: own, .. } if own == index) {
                        top.extra.self_loop = true;
                    }
                }
                return Reached::Answer(node, value);
            }
            None => {}
        }
        match fold.enter(&node) {
            // A leaf has no successors, so it lies on no cycle.
            FoldEnter::Leaf(value) => {
                self.slots.insert(mark, Slot::Done(value.clone()));
                Reached::Answer(node, value)
            }
            FoldEnter::Fold { accumulator, .. } => {
                let index = self.next_index;
                self.next_index += 1;
                self.slots
                    .insert(mark.clone(), Slot::Open { index, value: None });
                let position = self.members.len();
                self.members.push(Member {
                    mark,
                    node: None,
                    finished: 0,
                });
                let fix = Fix {
                    role: Role::Marked { index, position },
                    lowlink: index,
                    self_loop: false,
                };
                self.push_frame(fold, node, accumulator, fix);
                Reached::Opened
            }
        }
    }

    /// Hands a finished answer to the parent frame, or returns it as the
    /// root's.
    fn deliver(&mut self, fold: &mut F, node: &F::Node, value: Option<F::Value>) -> Step<F::Value> {
        if self.walk.stack.is_empty() {
            Step::Root(value)
        } else {
            self.walk.absorb(fold, node, value);
            Step::Continue
        }
    }

    /// Stores a member's current answer, reporting whether it changed.
    fn update(&mut self, position: usize, value: Option<F::Value>) -> bool {
        let slot = self.slots.get_mut(&self.members[position].mark);
        let Some(Slot::Open { value: current, .. }) = slot else {
            unreachable!("a member's slot stays open until its cycle is final");
        };
        let changed = *current != value;
        *current = value;
        changed
    }

    /// Handles a frame that finished folding.
    fn finish(
        &mut self,
        fold: &mut F,
        node: F::Node,
        value: Option<F::Value>,
        fix: Fix,
    ) -> Result<Step<F::Value>, OpenFixpointError> {
        if !matches!(fix.role, Role::Refold { .. })
            && let Some(parent) = self.walk.stack.last_mut()
        {
            parent.extra.lowlink = parent.extra.lowlink.min(fix.lowlink);
        }
        match fix.role {
            Role::Unmarked { entry } => {
                // The subtree reached a member that was open before this node
                // opened: that member reaches back here, so this node is on
                // its cycle.
                if fix.lowlink < entry {
                    return Err(OpenFixpointError::UnmarkedCycle);
                }
                Ok(self.deliver(fold, &node, value))
            }
            Role::Marked { index, position } => {
                if fix.lowlink == index && position + 1 == self.members.len() && !fix.self_loop {
                    // A node on no cycle: its first answer is final.
                    if let Some(member) = self.members.pop() {
                        self.slots.insert(member.mark, Slot::Done(value.clone()));
                    }
                    return Ok(self.deliver(fold, &node, value));
                }
                self.update(position, value.clone());
                self.members[position].finished = self.finished;
                self.finished += 1;
                if fix.lowlink < index {
                    // A member whose cycle starts further up: the parent
                    // absorbs its provisional answer.
                    self.walk.absorb(fold, &node, value);
                    self.members[position].node = Some(node);
                    return Ok(Step::Continue);
                }
                self.members[position].node = Some(node);
                self.components.push(Component {
                    base: position,
                    index,
                    lowlink: index,
                    order: Vec::new(),
                    next: 0,
                    rounds: 0,
                    changed: true,
                });
                self.advance(fold)
            }
            Role::Refold { position } => {
                let changed = self.update(position, value);
                self.members[position].node = Some(node);
                if let Some(component) = self.components.last_mut() {
                    component.lowlink = component.lowlink.min(fix.lowlink);
                    component.changed |= changed;
                    component.next += 1;
                }
                self.advance(fold)
            }
        }
    }

    /// Re-folds the next member of the innermost cycle, starts its next
    /// round, or finalizes it once a round changed nothing.
    fn advance(&mut self, fold: &mut F) -> Result<Step<F::Value>, OpenFixpointError> {
        loop {
            let Some(component) = self.components.last_mut() else {
                unreachable!("only a cycle being iterated advances");
            };
            if component.next == component.order.len() {
                if component.lowlink < component.index {
                    return Ok(self.merge_outward(fold));
                }
                // A member found in the last round joins the next one.
                let grew = self.members.len() != component.base + component.order.len();
                if !component.changed && !grew {
                    return Ok(self.finalize(fold));
                }
                if let Some(max) = self.config.max_scc_rounds
                    && component.rounds >= max
                {
                    return Err(OpenFixpointError::RoundLimit {
                        rounds: component.rounds,
                    });
                }
                component.rounds += 1;
                component.changed = false;
                component.next = 0;
                component.order.clear();
                component.order.extend(component.base..self.members.len());
                let members = &self.members;
                component
                    .order
                    .sort_unstable_by_key(|&position| members[position].finished);
            }
            let position = component.order[component.next];
            let Some(node) = self.members[position].node.take() else {
                unreachable!("a member waits outside any frame between its folds");
            };
            match fold.enter(&node) {
                FoldEnter::Leaf(value) => {
                    let changed = self.update(position, value);
                    self.members[position].node = Some(node);
                    if let Some(component) = self.components.last_mut() {
                        component.changed |= changed;
                        component.next += 1;
                    }
                }
                FoldEnter::Fold { accumulator, .. } => {
                    let fix = Fix {
                        role: Role::Refold { position },
                        lowlink: usize::MAX,
                        self_loop: false,
                    };
                    self.push_frame(fold, node, accumulator, fix);
                    return Ok(Step::Continue);
                }
            }
        }
    }

    /// Gives up the innermost cycle as a cycle of its own: a re-fold reached
    /// an older open node, so its members belong to that node's cycle. They
    /// stay open with their current answers, and the frame that reached the
    /// first member absorbs that member's answer, as it would absorb any
    /// member whose cycle starts further up.
    fn merge_outward(&mut self, fold: &mut F) -> Step<F::Value> {
        let Some(component) = self.components.pop() else {
            unreachable!("only a cycle being iterated merges");
        };
        let member = &mut self.members[component.base];
        let Some(node) = member.node.take() else {
            unreachable!("a cycle's first member waits outside any frame");
        };
        let value = match self.slots.get(&member.mark) {
            Some(Slot::Open { value, .. }) => value.clone(),
            _ => unreachable!("a member's slot stays open until its cycle is final"),
        };
        if let Some(parent) = self.walk.stack.last_mut() {
            parent.extra.lowlink = parent.extra.lowlink.min(component.lowlink);
        }
        self.walk.absorb(fold, &node, value);
        self.members[component.base].node = Some(node);
        Step::Continue
    }

    /// Makes every answer of the innermost cycle final and hands its first
    /// node's answer to the frame that reached it.
    fn finalize(&mut self, fold: &mut F) -> Step<F::Value> {
        let Some(component) = self.components.pop() else {
            unreachable!("only a cycle being iterated is finalized");
        };
        let mut root = None;
        for member in self.members.drain(component.base..) {
            let Some(slot) = self.slots.get_mut(&member.mark) else {
                unreachable!("every member has a slot");
            };
            if let Slot::Open { value, .. } = slot {
                let value = value.take();
                if root.is_none() {
                    root = Some((member.node, value.clone()));
                }
                *slot = Slot::Done(value);
            }
        }
        let Some((Some(node), value)) = root else {
            unreachable!("a cycle's first member waits outside any frame");
        };
        self.deliver(fold, &node, value)
    }
}

/// Folds the space reachable from `root` in post-order and iterates every
/// cycle of marked nodes until its answers are stable.
///
/// [`open_fold_post_order`](super::open_fold_post_order) refuses a node that
/// is still open on the route, so a fold over a cyclic space gets an answer
/// that misses what travels around the cycle and that depends on where the
/// walk entered it. This driver keeps one answer per [`mark`](OpenFold::mark)
/// instead: it finds the strongly connected components of marked nodes with
/// Tarjan's lowlinks as it walks, and re-folds the nodes of each cycle until
/// a round changes no answer. Then every node of the cycle has its least
/// fixpoint answer, whichever node the walk entered first.
///
/// - **The walk.** The frames are those of the plain driver. A marked node
///   that is still open absorbs its current, provisional answer, which
///   starts as `None`; a finished marked node absorbs its final answer. The
///   driver memoizes every marked node, so it reads neither
///   [`memo_key`](OpenFold::memo_key) nor the [`MarkScope`] of
///   [`FoldEnter::Fold`]. A node without a mark is folded on every reach.
/// - **Iteration.** When the first node of a cycle finishes, each member of
///   the cycle is re-folded in the order its first fold finished: the driver
///   calls [`enter`](OpenFold::enter), absorbs each successor's current answer
///   without descending into members, and calls
///   [`finish`](OpenFold::finish). Rounds repeat until no answer changes, and
///   the answers become final. Hooks with side effects observe a member's
///   hooks once per round.
/// - **The contract.** `enter`, `absorb` and `finish` must be deterministic,
///   and monotone in the absorbed answers over a domain of finite height; or
///   [`OpenFixpointConfig::with_max_scc_rounds`] bounds the rounds. A node on
///   a cycle needs a mark: a cycle that passes a marked node and a node
///   without a mark is reported as [`OpenFixpointError::UnmarkedCycle`]. A
///   cycle of nodes without marks is never closed, so only the depth bound
///   stops it.
///
/// On a space without cycles this is the walk of
/// [`open_fold_post_order_memoized`](super::open_fold_post_order_memoized)
/// with a memo key equal to each node's mark: `O(n + m)` hook calls. Each
/// cycle adds `O(rounds × (n_scc + m_scc))` hook calls. The successor order
/// is fixed, so results are deterministic.
///
/// # Errors
///
/// Returns [`OpenFixpointError::RoundLimit`] when a cycle still changes after
/// the configured rounds, and [`OpenFixpointError::UnmarkedCycle`] when a
/// node without a mark lies on a cycle with a marked node.
///
/// ```rust
/// use std::collections::BTreeSet;
/// use core::ops::ControlFlow;
/// use cfglib::{
///     FoldEnter, MarkScope, OpenFixpointConfig, OpenFold, OpenFoldConfig, open_fold_fixpoint,
///     open_fold_post_order,
/// };
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
/// let both = BTreeSet::from(["x", "z"]);
/// for module in [0, 1] {
///     let mut exports = Exports { own, reexports };
///     assert_eq!(
///         open_fold_fixpoint(&mut exports, module, OpenFixpointConfig::new()),
///         Ok(Some(both.clone())),
///     );
/// }
/// // One post-order pass refuses 0 under 1, so 1 brings nothing back to 0.
/// let mut exports = Exports { own, reexports };
/// assert_eq!(
///     open_fold_post_order(&mut exports, 0, OpenFoldConfig::new()),
///     Some(BTreeSet::from(["x"])),
/// );
/// ```
pub fn open_fold_fixpoint<F: OpenFold>(
    fold: &mut F,
    root: F::Node,
    config: OpenFixpointConfig,
) -> Result<Option<F::Value>, OpenFixpointError>
where
    F::Value: Clone + PartialEq,
{
    let mut run = Run::<F> {
        walk: Walk::new(),
        slots: BTreeMap::new(),
        members: Vec::new(),
        components: Vec::new(),
        next_index: 0,
        finished: 0,
        config,
    };
    if let Reached::Answer(_, value) = run.reach(fold, root) {
        return Ok(value);
    }
    loop {
        if let Some(child) = run.walk.next_child() {
            if !run.walk.admit_child(config.max_depth) {
                run.walk.absorb(fold, &child, None);
            } else if let Reached::Answer(child, value) = run.reach(fold, child) {
                run.walk.absorb(fold, &child, value);
            }
            continue;
        }
        let Some((node, value, fix)) = run.walk.finish_top(fold) else {
            unreachable!("the walk returns once its root's answer is final");
        };
        if let Step::Root(value) = run.finish(fold, node, value, fix)? {
            return Ok(value);
        }
    }
}
