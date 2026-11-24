//! Greedy chain merging for the Ext-TSP order.
//!
//! A chain is a run of local nodes that stay together, laid out from address
//! zero. Jumps between two chains score nothing until the chains merge, so
//! the gain of a merge is the score of the jumps between the two chains in
//! the merged chain, plus the change of the jumps of a split chain that cross
//! the split. The jumps on one side of the split keep their score: their
//! ends move together. Self-loops score the same in every order and take no
//! part.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use super::{ExtTspConfig, Jump, Problem, jump_score};

/// Orders the local nodes of `problem`, whose node 0 is the entry, by greedy
/// chain merging.
pub(super) fn merge(problem: &Problem, config: &ExtTspConfig) -> Vec<usize> {
    let mut merger = Merger::new(problem, config);
    merger.run();
    merger.order()
}

/// Where a merge puts chain `y` relative to chain `x`, which it cuts at
/// `split` into `x1 = x[..split]` and `x2 = x[split..]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Shape {
    /// `x1 y x2`; a split at the end of `x` concatenates `x y`.
    Inside,
    /// `y x2 x1`.
    Before,
    /// `x2 x1 y`.
    After,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Merge {
    x: usize,
    y: usize,
    split: usize,
    shape: Shape,
}

/// A merge in the queue. The queue pops the largest gain first, then the
/// smaller pair of heads; the merge is stale once either chain has merged
/// since it was evaluated.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    gain: u128,
    heads: Reverse<(usize, usize)>,
    versions: (u32, u32),
    merge: Merge,
}

struct Chain {
    nodes: Vec<usize>,
    /// The bytes of all its nodes.
    size: u64,
    /// The jumps between two nodes of this chain.
    internal: Vec<usize>,
    /// The chains that share a jump with this one.
    neighbors: BTreeSet<usize>,
    /// Counts the merges that this chain survived.
    version: u32,
}

/// One part of a merged arrangement: `chain[low..high]`, placed at
/// `position` and `address`.
#[derive(Clone, Copy)]
struct Slice {
    chain: usize,
    low: usize,
    high: usize,
    position: usize,
    address: u64,
}

/// A jump between the nodes at `low` and `high` of one chain, with its score
/// there. A split of the chain at `split` cuts it when
/// `low < split <= high`.
#[derive(Clone, Copy)]
struct Span {
    low: usize,
    high: usize,
    jump: usize,
    score: u128,
}

/// The spans of one chain, swept by ascending split.
#[derive(Default)]
struct Sweep {
    /// Every span of the chain, by ascending `low`.
    pending: Vec<Span>,
    /// The next span of `pending` to start.
    next: usize,
    /// The spans that the current split cuts.
    cut: Vec<Span>,
}

impl Sweep {
    fn start(&mut self, spans: impl Iterator<Item = Span>) {
        self.pending.clear();
        self.pending.extend(spans);
        self.pending.sort_unstable_by_key(|span| span.low);
        self.next = 0;
        self.cut.clear();
    }

    /// Moves to `split`, which grows from one call to the next.
    fn advance(&mut self, split: usize) {
        while let Some(&span) = self.pending.get(self.next)
            && span.low < split
        {
            self.cut.push(span);
            self.next += 1;
        }
        self.cut.retain(|span| split <= span.high);
    }
}

struct Merger<'p> {
    problem: &'p Problem,
    config: &'p ExtTspConfig,
    /// The live chains by identity; a chain merged away leaves `None`.
    chains: Vec<Option<Chain>>,
    /// The jumps between two chains, keyed by the ordered pair of identities.
    between: BTreeMap<(usize, usize), Vec<usize>>,
    chain_of: Vec<usize>,
    index_of: Vec<usize>,
    /// The address of each node within its chain.
    offset_of: Vec<u64>,
    queue: BinaryHeap<Candidate>,
    /// Scratch for the splits of one evaluation.
    sweep: Sweep,
    /// Scratch for the neighbors of a merged chain.
    neighbors: Vec<usize>,
}

/// The key of a pair of chains.
fn pair(first: usize, second: usize) -> (usize, usize) {
    (first.min(second), first.max(second))
}

impl<'p> Merger<'p> {
    /// One chain per node, and the merges of every pair that shares a jump.
    fn new(problem: &'p Problem, config: &'p ExtTspConfig) -> Self {
        let count = problem.sizes.len();
        let mut chains: Vec<Option<Chain>> = (0..count)
            .map(|node| {
                Some(Chain {
                    nodes: vec![node],
                    size: problem.sizes[node],
                    internal: Vec::new(),
                    neighbors: BTreeSet::new(),
                    version: 0,
                })
            })
            .collect();
        let mut between: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (index, jump) in problem.jumps.iter().enumerate() {
            if jump.source == jump.target {
                continue;
            }
            between
                .entry(pair(jump.source, jump.target))
                .or_default()
                .push(index);
            for (end, other) in [(jump.source, jump.target), (jump.target, jump.source)] {
                chains[end]
                    .as_mut()
                    .expect("every chain is live before the first merge")
                    .neighbors
                    .insert(other);
            }
        }
        let pairs: Vec<(usize, usize)> = between.keys().copied().collect();
        let mut merger = Self {
            problem,
            config,
            chains,
            between,
            chain_of: (0..count).collect(),
            index_of: vec![0; count],
            offset_of: vec![0; count],
            queue: BinaryHeap::new(),
            sweep: Sweep::default(),
            neighbors: Vec::new(),
        };
        for (first, second) in pairs {
            merger.enqueue(first, second);
        }
        merger
    }

    fn chain(&self, id: usize) -> &Chain {
        self.chains[id]
            .as_ref()
            .expect("a merge names only live chains")
    }

    /// Applies the best merge until no merge gains.
    fn run(&mut self) {
        while let Some(candidate) = self.queue.pop() {
            let Merge { x, y, .. } = candidate.merge;
            let current = |id: usize| self.chains[id].as_ref().map(|chain| chain.version);
            if (current(x), current(y)) != (Some(candidate.versions.0), Some(candidate.versions.1))
            {
                continue;
            }
            let merged = self.apply(candidate.merge);
            let mut neighbors = core::mem::take(&mut self.neighbors);
            neighbors.clear();
            neighbors.extend(self.chain(merged).neighbors.iter().copied());
            for &other in &neighbors {
                self.enqueue(merged, other);
            }
            self.neighbors = neighbors;
        }
    }

    /// Queues the better merge of the two orders of a pair, if it gains.
    fn enqueue(&mut self, first: usize, second: usize) {
        let mut sweep = core::mem::take(&mut self.sweep);
        let mut best: Option<(u128, Merge)> = None;
        for (x, y) in [(first, second), (second, first)] {
            if let Some((gain, merge)) = self.best_merge(x, y, &mut sweep)
                && best.is_none_or(|(best_gain, _)| gain > best_gain)
            {
                best = Some((gain, merge));
            }
        }
        self.sweep = sweep;
        if let Some((gain, merge)) = best {
            let (x, y) = (self.chain(merge.x), self.chain(merge.y));
            let candidate = Candidate {
                gain,
                heads: Reverse(pair(x.nodes[0], y.nodes[0])),
                versions: (x.version, y.version),
                merge,
            };
            self.queue.push(candidate);
        }
    }

    /// The merge of `y` into `x` with the largest positive gain, keeping the
    /// entry first.
    fn best_merge(&self, x: usize, y: usize, sweep: &mut Sweep) -> Option<(u128, Merge)> {
        let (x_chain, y_chain) = (self.chain(x), self.chain(y));
        let cross = self
            .between
            .get(&pair(x, y))
            .expect("neighboring chains share a jump");
        let entry_in_x = x_chain.nodes[0] == 0;
        let entry_in_y = y_chain.nodes[0] == 0;
        let mut best: Option<(u128, Merge)> = None;
        let mut consider = |gain: u128, merge: Merge| {
            if gain > 0 && best.is_none_or(|(best_gain, _)| gain > best_gain) {
                best = Some((gain, merge));
            }
        };

        let length = x_chain.nodes.len();
        if !entry_in_y {
            let merge = Merge {
                x,
                y,
                split: length,
                shape: Shape::Inside,
            };
            consider(self.gain(merge, cross, &[]), merge);
        }
        if length > 1 && length <= self.config.chain_split_threshold {
            sweep.start(x_chain.internal.iter().map(|&jump| self.span(jump)));
            for split in 1..length {
                sweep.advance(split);
                for shape in [Shape::Inside, Shape::Before, Shape::After] {
                    let keeps_entry_first = match shape {
                        Shape::Inside => !entry_in_y,
                        Shape::Before => !entry_in_x,
                        Shape::After => !entry_in_x && !entry_in_y,
                    };
                    if keeps_entry_first {
                        let merge = Merge { x, y, split, shape };
                        consider(self.gain(merge, cross, &sweep.cut), merge);
                    }
                }
            }
        }
        best
    }

    /// The score that `merge` adds: the jumps of `cross` in the merged chain,
    /// and the change of the jumps that the split cuts.
    fn gain(&self, merge: Merge, cross: &[usize], cut: &[Span]) -> u128 {
        let slices = self.slices(merge);
        let added = cross
            .iter()
            .chain(cut.iter().map(|span| &span.jump))
            .map(|&jump| self.merged_score(&slices, jump))
            .fold(0, u128::saturating_add);
        let lost = cut
            .iter()
            .map(|span| span.score)
            .fold(0, u128::saturating_add);
        added.saturating_sub(lost)
    }

    /// The span of `jump`, whose ends are in one chain.
    fn span(&self, jump: usize) -> Span {
        let Jump { source, target, .. } = self.problem.jumps[jump];
        let (source_index, target_index) = (self.index_of[source], self.index_of[target]);
        Span {
            low: source_index.min(target_index),
            high: source_index.max(target_index),
            jump,
            score: self.chain_score(jump),
        }
    }

    /// The parts of the arrangement that `merge` makes, in order.
    fn slices(&self, merge: Merge) -> [Slice; 3] {
        let x_len = self.chain(merge.x).nodes.len();
        let y_len = self.chain(merge.y).nodes.len();
        let (x, y, split) = (merge.x, merge.y, merge.split);
        let parts = match merge.shape {
            Shape::Inside => [(x, 0, split), (y, 0, y_len), (x, split, x_len)],
            Shape::Before => [(y, 0, y_len), (x, split, x_len), (x, 0, split)],
            Shape::After => [(x, split, x_len), (x, 0, split), (y, 0, y_len)],
        };
        let (mut position, mut address) = (0, 0_u64);
        parts.map(|(chain, low, high)| {
            let slice = Slice {
                chain,
                low,
                high,
                position,
                address,
            };
            position += high - low;
            address =
                address.saturating_add(self.offset_at(chain, high) - self.offset_at(chain, low));
            slice
        })
    }

    /// The address within `chain` where its node at `index` starts, or its
    /// size at the end.
    fn offset_at(&self, chain: usize, index: usize) -> u64 {
        let chain = self.chain(chain);
        chain
            .nodes
            .get(index)
            .map_or(chain.size, |&node| self.offset_of[node])
    }

    /// The position and address of `node` in an arrangement.
    fn place(&self, slices: &[Slice; 3], node: usize) -> (usize, u64) {
        let (chain, index) = (self.chain_of[node], self.index_of[node]);
        let slice = slices
            .iter()
            .find(|slice| slice.chain == chain && (slice.low..slice.high).contains(&index))
            .expect("a jump of the merged chains has both ends in them");
        (
            slice.position + index - slice.low,
            slice
                .address
                .saturating_add(self.offset_of[node] - self.offset_at(chain, slice.low)),
        )
    }

    /// The score of jump `index` in the arrangement of `slices`.
    fn merged_score(&self, slices: &[Slice; 3], index: usize) -> u128 {
        let jump = &self.problem.jumps[index];
        let (source_position, source_address) = self.place(slices, jump.source);
        let (target_position, target_address) = self.place(slices, jump.target);
        jump_score(
            jump.count,
            jump.falls && target_position == source_position + 1,
            source_address.saturating_add(self.problem.sizes[jump.source]),
            target_address,
            self.config,
        )
    }

    /// The score of jump `index` within the chain that holds both its ends.
    fn chain_score(&self, index: usize) -> u128 {
        let jump = &self.problem.jumps[index];
        jump_score(
            jump.count,
            jump.falls && self.index_of[jump.target] == self.index_of[jump.source] + 1,
            self.offset_of[jump.source].saturating_add(self.problem.sizes[jump.source]),
            self.offset_of[jump.target],
            self.config,
        )
    }

    /// Merges the two chains of `merge` into the one with the smaller
    /// identity, and returns it.
    fn apply(&mut self, merge: Merge) -> usize {
        let Merge { x, y, split, shape } = merge;
        let x_chain = self.chains[x].take().expect("a merge names live chains");
        let y_chain = self.chains[y].take().expect("a merge names live chains");
        let survivor = x.min(y);

        let (x1, x2) = x_chain.nodes.split_at(split);
        let y_nodes = y_chain.nodes.as_slice();
        let nodes = match shape {
            Shape::Inside => [x1, y_nodes, x2],
            Shape::Before => [y_nodes, x2, x1],
            Shape::After => [x2, x1, y_nodes],
        }
        .concat();
        let mut size = 0_u64;
        for (index, &node) in nodes.iter().enumerate() {
            self.chain_of[node] = survivor;
            self.index_of[node] = index;
            self.offset_of[node] = size;
            size = size.saturating_add(self.problem.sizes[node]);
        }

        let mut internal = x_chain.internal;
        internal.extend(y_chain.internal);
        internal.extend(self.between.remove(&pair(x, y)).unwrap_or_default());
        let mut neighbors = BTreeSet::new();
        for (old, old_neighbors) in [(x, x_chain.neighbors), (y, y_chain.neighbors)] {
            for other in old_neighbors {
                if other == x || other == y {
                    continue;
                }
                let jumps = self
                    .between
                    .remove(&pair(old, other))
                    .expect("neighboring chains share a jump");
                self.between
                    .entry(pair(survivor, other))
                    .or_default()
                    .extend(jumps);
                let other_chain = self.chains[other]
                    .as_mut()
                    .expect("a neighbor of a live chain is live");
                other_chain.neighbors.remove(&old);
                other_chain.neighbors.insert(survivor);
                neighbors.insert(other);
            }
        }

        self.chains[survivor] = Some(Chain {
            nodes,
            size,
            internal,
            neighbors,
            version: x_chain.version.max(y_chain.version) + 1,
        });
        survivor
    }

    /// The live chains by their first node: the entry's chain first.
    fn order(self) -> Vec<usize> {
        let mut chains: Vec<Chain> = self.chains.into_iter().flatten().collect();
        chains.sort_unstable_by_key(|chain| chain.nodes[0]);
        chains.into_iter().flat_map(|chain| chain.nodes).collect()
    }
}
