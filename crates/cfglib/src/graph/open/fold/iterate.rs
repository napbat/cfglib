//! The post-order fold that iterates each cycle of marked nodes to a
//! fixpoint.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::walk::Walk;
use super::{FoldEnter, MarkScope, OpenFold, OpenFoldError};

#[cfg(test)]
mod tests;

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

/// The provisional answer of a member whose cycle is not finished.
struct Provisional<V> {
    /// The member's index in discovery order.
    index: usize,
    /// Its current answer.
    value: Option<V>,
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

struct Run<'a, F: OpenFold> {
    walk: Walk<F, Fix>,
    /// The final answers by mark: the caller's table.
    answers: &'a mut BTreeMap<F::Mark, Option<F::Value>>,
    /// The provisional answers of the members by mark. They stay out of the
    /// caller's table, so it never holds an answer that is not final.
    provisional: BTreeMap<F::Mark, Provisional<F::Value>>,
    members: Vec<Member<F>>,
    components: Vec<Component>,
    next_index: usize,
    finished: usize,
    max_rounds: Option<usize>,
}

impl<F: OpenFold> Run<'_, F>
where
    F::Value: Clone + PartialEq,
{
    fn push_frame(&mut self, fold: &mut F, node: F::Node, accumulator: F::Accumulator, fix: Fix) {
        // Every marked node is memoized, so no mark scope applies.
        self.walk
            .push_frame(fold, node, accumulator, MarkScope::Shared, 0, fix);
    }

    /// Answers `node` from its final or provisional answer or as a leaf, or
    /// opens a frame for it.
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
        if let Some(value) = self.answers.get(&mark) {
            return Reached::Answer(node, value.clone());
        }
        if let Some(slot) = self.provisional.get(&mark) {
            let index = slot.index;
            let value = slot.value.clone();
            if let Some(top) = self.walk.stack.last_mut() {
                top.extra.lowlink = top.extra.lowlink.min(index);
                if matches!(top.extra.role, Role::Marked { index: own, .. } if own == index) {
                    top.extra.self_loop = true;
                }
            }
            return Reached::Answer(node, value);
        }
        match fold.enter(&node) {
            // A leaf has no successors, so it lies on no cycle.
            FoldEnter::Leaf(value) => {
                self.answers.insert(mark, value.clone());
                Reached::Answer(node, value)
            }
            FoldEnter::Fold { accumulator, .. } => {
                let index = self.next_index;
                self.next_index += 1;
                self.provisional
                    .insert(mark.clone(), Provisional { index, value: None });
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
        let Some(slot) = self.provisional.get_mut(&self.members[position].mark) else {
            unreachable!("a member's answer stays provisional until its cycle is final");
        };
        let changed = slot.value != value;
        slot.value = value;
        changed
    }

    /// Handles a frame that finished folding.
    fn finish(
        &mut self,
        fold: &mut F,
        node: F::Node,
        value: Option<F::Value>,
        fix: Fix,
    ) -> Result<Step<F::Value>, OpenFoldError> {
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
                    return Err(OpenFoldError::UnmarkedCycle);
                }
                Ok(self.deliver(fold, &node, value))
            }
            Role::Marked { index, position } => {
                if fix.lowlink == index && position + 1 == self.members.len() && !fix.self_loop {
                    // A node on no cycle: its first answer is final.
                    if let Some(member) = self.members.pop() {
                        self.provisional.remove(&member.mark);
                        self.answers.insert(member.mark, value.clone());
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
    fn advance(&mut self, fold: &mut F) -> Result<Step<F::Value>, OpenFoldError> {
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
                if let Some(limit) = self.max_rounds
                    && component.rounds >= limit
                {
                    return Err(OpenFoldError::RoundLimitExceeded {
                        limit,
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
        let Some(slot) = self.provisional.get(&member.mark) else {
            unreachable!("a member's answer stays provisional until its cycle is final");
        };
        let value = slot.value.clone();
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
            let Some(slot) = self.provisional.remove(&member.mark) else {
                unreachable!("a member's answer stays provisional until its cycle is final");
            };
            if root.is_none() {
                root = Some((member.node, slot.value.clone()));
            }
            self.answers.insert(member.mark, slot.value);
        }
        let Some((Some(node), value)) = root else {
            unreachable!("a cycle's first member waits outside any frame");
        };
        self.deliver(fold, &node, value)
    }
}

/// Folds the space reachable from `root` in post-order and iterates every
/// cycle of marked nodes until its answers are stable: `answers` holds the
/// final answers by mark, and the walk reads and extends it.
pub(super) fn run<F: OpenFold>(
    answers: &mut BTreeMap<F::Mark, Option<F::Value>>,
    fold: &mut F,
    root: F::Node,
    max_depth: Option<usize>,
    max_rounds: Option<usize>,
) -> Result<Option<F::Value>, OpenFoldError>
where
    F::Value: Clone + PartialEq,
{
    let mut run = Run::<F> {
        walk: Walk::new(),
        answers,
        provisional: BTreeMap::new(),
        members: Vec::new(),
        components: Vec::new(),
        next_index: 0,
        finished: 0,
        max_rounds,
    };
    if let Reached::Answer(_, value) = run.reach(fold, root) {
        return Ok(value);
    }
    loop {
        if let Some(child) = run.walk.next_child() {
            if !run.walk.admit_child(max_depth) {
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
