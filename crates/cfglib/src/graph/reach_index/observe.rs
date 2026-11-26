//! The constant-time observations of a [`ReachIndex`](super::ReachIndex) and
//! the pruned bidirectional search behind them.
//!
//! Every observation is a necessary condition of reachability (a violated
//! one refutes a pair) or a witness of it (a satisfied one proves the pair),
//! so the answers they give are exact, and a pair neither refuted nor proven
//! is searched.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use super::{ReachIndexConfig, ReachScratch};
use crate::graph::fanout::DenseFanout;
use crate::identity::{NO_INDEX, word};
use crate::union_find::DisjointSet;

/// The condensation, its observations, and nothing else.
///
/// Components are numbered sources first, so every condensation edge leads
/// from a lower to a higher component index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Observations {
    /// Per component, the components its edges lead to.
    successors: DenseFanout<u32>,
    /// Per component, the components whose edges lead to it.
    predecessors: DenseFanout<u32>,
    /// Per component, the observations every configuration keeps.
    facts: Vec<Facts>,
    /// How many orders `orders` holds per component.
    order_count: usize,
    /// Per component and order, component-major, so the orders of one
    /// component sit side by side: component `c` in order `i` is entry
    /// `c * order_count + i`.
    orders: Vec<OrderEntry>,
    /// Per component, its reach to and from the supportive components; empty
    /// when there are none.
    support: Vec<SupportRow>,
}

/// The observations every component has, whatever the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Facts {
    /// The weakly connected component, as a representative member: two
    /// components of different weak components never reach each other.
    weak: u32,
    /// The longest path from a source component to this one, which grows
    /// along every edge.
    forward_level: u32,
    /// The longest path from this component to a sink component, which
    /// shrinks along every edge.
    backward_level: u32,
}

/// One component's place in one extended depth-first topological order.
///
/// The order is a depth-first postorder: reversed for a walk along the edges,
/// as it is for a walk against them. Either way it is topological, and each
/// component's depth-first subtree occupies one contiguous run of positions
/// next to the component itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct OrderEntry {
    /// Position in the order.
    position: u32,
    /// The far end of the component's subtree run: its highest position for a
    /// walk along the edges, where the run follows the component and holds
    /// components it reaches, and its lowest position for a walk against them,
    /// where the run precedes the component and holds components that reach
    /// it.
    tree: u32,
    /// The highest position of any component this one reaches, itself
    /// included.
    max_reached: u32,
    /// The lowest position of any component that reaches this one, itself
    /// included.
    min_reaching: u32,
}

/// One component's reach to and from the supportive components, one bit each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct SupportRow {
    /// Bit `j`: this component reaches supportive component `j`.
    reaches: u64,
    /// Bit `j`: supportive component `j` reaches this component.
    reached_by: u64,
}

impl Observations {
    /// Compute the observations of the condensation with these adjacency
    /// lists under `config`.
    pub(super) fn compute(
        successors: DenseFanout<u32>,
        predecessors: DenseFanout<u32>,
        config: ReachIndexConfig,
    ) -> Self {
        let facts = facts(&successors, &predecessors);
        let orders = orders(&successors, &predecessors, config.orders);
        let support = support(&successors, &predecessors, &facts, config.supportive);
        Self {
            successors,
            predecessors,
            facts,
            order_count: config.orders,
            orders,
            support,
        }
    }

    /// Whether component `source` reaches component `target`, numbered
    /// before it: by observation when one decides the pair, by search
    /// otherwise.
    pub(super) fn reaches(&self, source: usize, target: usize, scratch: &mut ReachScratch) -> bool {
        if let Some(answer) = self.decide(source, target) {
            return answer;
        }
        #[cfg(test)]
        {
            scratch.coverage.searched += 1;
        }
        self.search(source, target, scratch)
    }

    /// Decide whether component `source` reaches component `target` by
    /// observation alone, or `None` when no observation does.
    fn decide(&self, source: usize, target: usize) -> Option<bool> {
        match source.cmp(&target) {
            Ordering::Equal => return Some(true),
            Ordering::Greater => return Some(false),
            Ordering::Less => {}
        }
        let (from, to) = (self.facts[source], self.facts[target]);
        if from.weak != to.weak
            || from.forward_level >= to.forward_level
            || from.backward_level <= to.backward_level
        {
            return Some(false);
        }
        if let (Some(from), Some(to)) = (self.support.get(source), self.support.get(target)) {
            if from.reaches & to.reached_by != 0 {
                return Some(true);
            }
            // A supportive component that reaches the source reaches the
            // target too, and one the target reaches the source reaches too.
            if from.reached_by & !to.reached_by != 0 || to.reaches & !from.reaches != 0 {
                return Some(false);
            }
        }
        let width = self.order_count;
        let from_orders = &self.orders[source * width..(source + 1) * width];
        let to_orders = &self.orders[target * width..(target + 1) * width];
        for (order, (from, to)) in from_orders.iter().zip(to_orders).enumerate() {
            if from.position > to.position
                || to.position > from.max_reached
                || from.position < to.min_reaching
            {
                return Some(false);
            }
            let in_subtree = if walks_along(order) {
                to.position <= from.tree
            } else {
                from.position >= to.tree
            };
            if in_subtree {
                return Some(true);
            }
        }
        None
    }

    /// Search for a path from component `source` to component `target`,
    /// alternately forward from the source and backward from the target,
    /// always expanding the side with fewer components waiting.
    ///
    /// Every component the forward side meets is reachable from the source,
    /// and every one the backward side meets reaches the target, so the two
    /// meeting proves the pair. A met component is first put to the
    /// observations against the far end: proven ends the search, refuted is
    /// not expanded. Neither prunes a component on a path from the source to
    /// the target, so a side that runs dry without meeting the other proves
    /// there is none.
    fn search(&self, source: usize, target: usize, scratch: &mut ReachScratch) -> bool {
        let ReachScratch {
            marks,
            forward,
            backward,
            ..
        } = scratch;
        marks.grow_to(2 * self.facts.len());
        marks.reset();
        forward.clear();
        backward.clear();
        marks.mark(2 * source);
        forward.push(word(source));
        marks.mark(2 * target + 1);
        backward.push(word(target));

        let (mut forward_head, mut backward_head) = (0, 0);
        while forward_head < forward.len() && backward_head < backward.len() {
            if forward.len() - forward_head <= backward.len() - backward_head {
                let component = forward[forward_head] as usize;
                forward_head += 1;
                for &next in self.successors.get(component) {
                    let next = next as usize;
                    if marks.is_marked(2 * next + 1) {
                        return true;
                    }
                    if marks.is_marked(2 * next) {
                        continue;
                    }
                    marks.mark(2 * next);
                    match self.decide(next, target) {
                        Some(true) => return true,
                        Some(false) => {}
                        None => forward.push(word(next)),
                    }
                }
            } else {
                let component = backward[backward_head] as usize;
                backward_head += 1;
                for &previous in self.predecessors.get(component) {
                    let previous = previous as usize;
                    if marks.is_marked(2 * previous) {
                        return true;
                    }
                    if marks.is_marked(2 * previous + 1) {
                        continue;
                    }
                    marks.mark(2 * previous + 1);
                    match self.decide(source, previous) {
                        Some(true) => return true,
                        Some(false) => {}
                        None => backward.push(word(previous)),
                    }
                }
            }
        }
        false
    }
}

/// Whether order `order` walks along the edges; the odd orders walk against
/// them.
const fn walks_along(order: usize) -> bool {
    order.is_multiple_of(2)
}

/// The weak component ids and the two topological levels.
///
/// Sources first means ascending component order is topological, so each
/// level is one pass over the adjacency lists in the right direction, with
/// the levels it reads already final.
fn facts(successors: &DenseFanout<u32>, predecessors: &DenseFanout<u32>) -> Vec<Facts> {
    let count = successors.bound();
    let mut facts = vec![Facts::default(); count];
    let mut weak = DisjointSet::new(count);
    for component in 0..count {
        let mut level = 0;
        for &previous in predecessors.get(component) {
            weak.union(component, previous as usize);
            level = level.max(facts[previous as usize].forward_level + 1);
        }
        facts[component].forward_level = level;
    }
    for component in (0..count).rev() {
        let level = successors
            .get(component)
            .iter()
            .map(|&next| facts[next as usize].backward_level + 1)
            .max()
            .unwrap_or(0);
        let entry = &mut facts[component];
        entry.backward_level = level;
        entry.weak = word(weak.find(component));
    }
    facts
}

/// Compute `order_count` extended depth-first topological orders.
///
/// Order `i` walks along the edges when `i` is even and against them when it
/// is odd, starting from the lowest-numbered component along them (a source)
/// and from the highest against them (a sink). Adjacency is read in stored
/// order when `i / 2` is even and in reverse when it is odd, and each list is
/// entered `i / 4` places later, wrapping around. These fixed alternatives
/// stand in for O'Reach's random tie-breaking, so the orders, and with them
/// the index, are a function of the graph alone.
fn orders(
    successors: &DenseFanout<u32>,
    predecessors: &DenseFanout<u32>,
    order_count: usize,
) -> Vec<OrderEntry> {
    let count = successors.bound();
    let mut entries = vec![
        OrderEntry::default();
        count
            .checked_mul(order_count)
            .expect("the order table's size fits in usize")
    ];
    let Some(last) = count.checked_sub(1).map(word) else {
        return entries;
    };
    // Per component, the number of components finished when it was entered:
    // the first postorder number of its subtree.
    let mut entered = vec![NO_INDEX; count];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for order in 0..order_count {
        let along = walks_along(order);
        let adjacency = if along { successors } else { predecessors };
        let reversed = order / 2 % 2 == 1;
        let rotation = order / 4;
        let slot = |component: usize| component * order_count + order;

        entered.fill(NO_INDEX);
        let mut finished = 0_u32;
        for step in 0..count {
            let root = if along { step } else { count - 1 - step };
            if entered[root] != NO_INDEX {
                continue;
            }
            entered[root] = finished;
            stack.push((root, 0));
            while let Some((component, taken)) = stack.last_mut() {
                let component = *component;
                if let Some(next) = pick(adjacency.get(component), *taken, reversed, rotation) {
                    *taken += 1;
                    if entered[next] == NO_INDEX {
                        entered[next] = finished;
                        stack.push((next, 0));
                    }
                    continue;
                }
                stack.pop();
                // The subtree's postorder numbers are `entered..=finished`.
                // Reversed for a walk along the edges, they become positions
                // `last - finished..=last - entered`.
                let entry = &mut entries[slot(component)];
                if along {
                    entry.position = last - finished;
                    entry.tree = last - entered[component];
                } else {
                    entry.position = finished;
                    entry.tree = entered[component];
                }
                finished += 1;
            }
        }

        // Reach bounds, each over a direction whose bounds it reads are
        // already final in this component order.
        for component in (0..count).rev() {
            let mut max_reached = entries[slot(component)].position;
            for &next in successors.get(component) {
                max_reached = max_reached.max(entries[slot(next as usize)].max_reached);
            }
            entries[slot(component)].max_reached = max_reached;
        }
        for component in 0..count {
            let mut min_reaching = entries[slot(component)].position;
            for &previous in predecessors.get(component) {
                min_reaching = min_reaching.min(entries[slot(previous as usize)].min_reaching);
            }
            entries[slot(component)].min_reaching = min_reaching;
        }
    }
    entries
}

/// The neighbor a walk takes after `taken` others, or `None` once it has
/// taken them all.
fn pick(neighbors: &[u32], taken: usize, reversed: bool, rotation: usize) -> Option<usize> {
    let len = neighbors.len();
    if taken == len {
        return None;
    }
    let place = (taken + rotation % len) % len;
    let place = if reversed { len - 1 - place } else { place };
    Some(neighbors[place] as usize)
}

/// Choose up to `supportive` components and record every component's reach
/// to and from each of them.
///
/// The score favors balanced reach into and out of a component. Its first
/// key is the product of in- and out-degree in the condensation, plus one
/// each: a product is large only when both sides are, and a component that
/// merged a large cycle collects the edges of all its members. The second is
/// the smaller of the two levels, a lower bound on both the components a
/// component reaches and the ones that reach it; the last is the lower
/// component index. Both directions of reach are one pass of `u64` unions
/// over the adjacency lists, so the whole table costs O(n + m) however many
/// components are chosen.
fn support(
    successors: &DenseFanout<u32>,
    predecessors: &DenseFanout<u32>,
    facts: &[Facts],
    supportive: usize,
) -> Vec<SupportRow> {
    let count = successors.bound();
    let chosen = supportive.min(count);
    if chosen == 0 {
        return Vec::new();
    }
    let score = |component: usize| {
        let Facts {
            forward_level,
            backward_level,
            ..
        } = facts[component];
        let degree = (successors.get(component).len() + 1)
            .saturating_mul(predecessors.get(component).len() + 1);
        (degree, forward_level.min(backward_level))
    };
    let mut candidates: Vec<usize> = (0..count).collect();
    candidates.select_nth_unstable_by(chosen - 1, |&left, &right| {
        score(right).cmp(&score(left)).then(left.cmp(&right))
    });
    candidates.truncate(chosen);
    candidates.sort_unstable();

    let mut rows = vec![SupportRow::default(); count];
    for (bit, &component) in candidates.iter().enumerate() {
        let mask = 1_u64 << bit;
        rows[component].reaches |= mask;
        rows[component].reached_by |= mask;
    }
    for component in 0..count {
        for &previous in predecessors.get(component) {
            rows[component].reached_by |= rows[previous as usize].reached_by;
        }
    }
    for component in (0..count).rev() {
        for &next in successors.get(component) {
            rows[component].reaches |= rows[next as usize].reaches;
        }
    }
    rows
}
