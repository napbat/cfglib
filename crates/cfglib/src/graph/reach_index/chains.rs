//! The exact chain mode: a chain cover of the condensation with, per
//! component and chain, the earliest member the component reaches.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::fanout::DenseFanout;
use crate::identity::word;

/// A greedy chain cover of the condensation and the reach of every component
/// into every chain.
///
/// A chain lists components that reach one another in order, so reaching one
/// member means reaching every member after it. A member's **depth** counts
/// the members after it, which makes "the earliest member reached" the
/// largest depth reached, and `source` reaches `target` exactly when
/// `source`'s deepest member of `target`'s chain is at least as deep as
/// `target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ChainTable {
    /// Per component, the chain it belongs to.
    chain: Vec<u32>,
    /// Per component, how many members of its chain come after it.
    depth: Vec<u32>,
    /// Row bounds into `reach`, indexed by build step: the row of component
    /// `c` is `offsets[n - 1 - c]..offsets[n - c]`.
    offsets: Vec<usize>,
    /// Per component and chain, one more than the deepest member the
    /// component reaches, zero when it reaches none. A row covers the chains
    /// that existed when its component was placed; the chains opened later
    /// hold only lower-numbered components, which it cannot reach.
    reach: Vec<u32>,
    /// How many chains the cover has.
    chain_count: usize,
}

impl ChainTable {
    /// Cover the condensation whose components, numbered sources first, have
    /// `successors`, unless that takes more than `max_chains` chains.
    ///
    /// Components are placed in descending order, so every successor already
    /// has its row: a component's row is the maximum of its successors' rows.
    /// Among the chains whose first member the component reaches, it joins
    /// the one whose first member is numbered lowest, and opens a new chain
    /// when it reaches none. That is O(chains) per component and per edge.
    pub(super) fn compute(successors: &DenseFanout<u32>, max_chains: usize) -> Option<Self> {
        let count = successors.bound();
        let mut chain = vec![0_u32; count];
        let mut depth = vec![0_u32; count];
        // Per chain, its current first member.
        let mut heads: Vec<u32> = Vec::new();
        let mut offsets = Vec::with_capacity(count + 1);
        offsets.push(0);
        let mut reach: Vec<u32> = Vec::new();

        for component in (0..count).rev() {
            let start = reach.len();
            reach.resize(start + heads.len(), 0);
            let (placed, row) = reach.split_at_mut(start);
            for &next in successors.get(component) {
                let step = count - 1 - next as usize;
                let theirs = &placed[offsets[step]..offsets[step + 1]];
                for (mine, &their) in row.iter_mut().zip(theirs) {
                    *mine = (*mine).max(their);
                }
            }

            let joined = heads
                .iter()
                .zip(row.iter())
                .enumerate()
                .filter(|&(_, (&head, &reached))| reached == depth[head as usize] + 1)
                .min_by_key(|&(_, (&head, _))| head)
                .map(|(index, _)| index);
            if let Some(index) = joined {
                let member_depth = depth[heads[index] as usize] + 1;
                depth[component] = member_depth;
                chain[component] = word(index);
                heads[index] = word(component);
                row[index] = member_depth + 1;
            } else {
                if heads.len() == max_chains {
                    return None;
                }
                chain[component] = word(heads.len());
                heads.push(word(component));
                reach.push(1);
            }
            offsets.push(reach.len());
        }

        Some(Self {
            chain,
            depth,
            offsets,
            reach,
            chain_count: heads.len(),
        })
    }

    /// Whether component `source` reaches component `target`.
    pub(super) fn reaches(&self, source: usize, target: usize) -> bool {
        let step = self.depth.len() - 1 - source;
        let row = &self.reach[self.offsets[step]..self.offsets[step + 1]];
        row.get(self.chain[target] as usize)
            .is_some_and(|&reached| reached > self.depth[target])
    }

    /// How many chains the cover has.
    pub(super) fn chain_count(&self) -> usize {
        self.chain_count
    }
}
