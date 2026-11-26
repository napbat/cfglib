//! Havlak's cycle nesting forest, in near-linear time.
//!
//! A depth-first walk from the root, which reads the successors of a block in
//! adjacency order, defines the forest. A block `w` heads a cycle when an edge
//! returns to it from itself or from below it in the walk's tree, and the
//! cycle is the blocks below `w` that reach `w` without leaving its subtree.
//! Those are the cycles of Havlak (TOPLAS 1997) and of LLVM's `CycleInfo`: the
//! outermost ones are the strongly connected components with an edge inside,
//! each headed by the block of it the walk visits first, and the cycles nested
//! in one are the outermost cycles of what is left of it without its header.
//! Two cycles are disjoint or one holds the other: when the cycle of `w` holds
//! a block of the cycle of a block `y` below `w`, it holds `y`, which reaches
//! that block inside its own subtree, and so every block that reaches `y`.
//!
//! The cycles are found innermost first, by visiting the blocks in reverse
//! preorder and walking back from the edges that return to each one, as
//! Havlak does, with a union-find that collapses every finished cycle into its
//! header so that a walk crosses it in one step. Havlak files an edge that
//! enters a cycle from outside under the cycle's header, where the walk of
//! every cycle around it takes the edge up and files it again, which costs
//! the nesting depth per edge. Following Ramalingam (TOPLAS 2002), an edge is
//! filed once instead, under the nearest common ancestor of its ends in the
//! walk's tree. No cycle headed below that block holds both ends, so the edge
//! waits there, and the visit of that block hands it to the cycle its target
//! has been collapsed into by then. Tarjan's offline algorithm finds the
//! ancestor during the walk itself, so every block and edge costs a constant
//! number of union-find steps: O(m α(n)) in all.
//!
//! The hand-off also finds the entries. The cycles that hold the target of an
//! edge but not its source are the ones headed below the ancestor, so the
//! target enters each cycle from the innermost around it to the one it has
//! been collapsed into, except the one it heads. A block shared by deeply
//! nested cycles enters every one of them, and a list per cycle would hold it
//! once for each, which is quadratic on deep irreducible nests. So each block
//! keeps only the outermost cycle it enters, and a difference over the forest
//! counts the entries of every cycle.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{LoopForest, LoopRecord};
use crate::graph::view::{DenseId, RootedView};
use crate::identity::{NO_INDEX, slot, word};
use crate::union_find::DisjointSet;

impl<N: DenseId> LoopForest<N> {
    /// Find every cycle of the code the root of `graph` reaches, whether or
    /// not an edge is tagged: Havlak's cycle nesting forest.
    ///
    /// A depth-first walk from the root, which reads the successors of each
    /// block in adjacency order, defines the forest. The outermost loops are
    /// the strongly connected components with an edge inside, each headed by
    /// the block of it the walk visits first, and the loops nested in a loop
    /// are the outermost loops of its blocks without its header. An
    /// irreducible cycle is therefore headed by the entry the walk reaches
    /// first, as in LLVM's `CycleInfo`, which is the same block every time for
    /// the same graph; [`entries`](Self::entries) lists the others.
    /// [`EdgeKind::Back`](crate::EdgeKind::Back) tags play no part.
    ///
    /// On a reducible graph every cycle is a natural loop, and the forest holds
    /// the loops of [`compute`](Self::compute). Every loop is
    /// [nested](Self::is_nested), irreducible or not, so
    /// [`contains`](Self::contains) and [`encloses`](Self::encloses) are
    /// constant time for all of them and [`parent`](Self::parent) is
    /// [`enclosing`](Self::enclosing). The [`latches`](Self::latches) of a loop
    /// are the tails of its edges into its header.
    ///
    /// The cost is near-linear in the blocks and edges, whatever the nesting,
    /// and the walk needs no recursion.
    ///
    /// # Panics
    ///
    /// Panics when the root is not below the node bound of `graph`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, LoopForest};
    ///
    /// // entry branches into `a` and `b`, which jump to each other: one cycle
    /// // with two entries, and no tag to say so.
    /// let mut cfg = Cfg::<u32>::new();
    /// let a = cfg.new_block();
    /// let b = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(a, b, EdgeKind::Jump);
    /// cfg.add_edge(b, a, EdgeKind::Jump);
    ///
    /// let forest = LoopForest::compute_cycles(&cfg);
    ///
    /// assert_eq!(forest.len(), 1);
    /// let cycle = forest.loop_of_header(a).expect("the walk reaches `a` first");
    /// assert_eq!(forest.entries(cycle), [a, b]);
    /// assert!(forest.is_irreducible(cycle));
    /// assert_eq!(forest.latches(cycle), [b]);
    /// assert!(forest.contains(cycle, b));
    /// ```
    #[must_use]
    pub fn compute_cycles<G: RootedView<NodeId = N>>(graph: &G) -> Self {
        let mut walk = Walk::new(graph);
        let cycles = Cycles::collapse(&mut walk);
        cycles.place(&walk, graph.node_bound())
    }
}

/// The depth-first walk that defines the forest, numbering the blocks it
/// visits in preorder, with every edge between them filed for the collapse.
struct Walk<N> {
    /// The block at each position of the preorder.
    blocks: Vec<N>,
    /// The position each edge leaves.
    sources: Vec<u32>,
    /// The position each edge enters.
    targets: Vec<u32>,
    /// The next edge of the list each edge is filed in.
    next: Vec<u32>,
    /// The first edge into each position from that position or below it.
    returning: Vec<u32>,
    /// The first other edge filed under each position: the edges whose ends
    /// have that position as nearest common ancestor.
    waiting: Vec<u32>,
}

/// A block of the walk with successors left to examine.
struct Frame {
    position: u32,
    /// Where the block's successors start in the arena; leaving the block
    /// truncates the arena back to it.
    start: usize,
    /// The next successor to examine, as an arena index.
    cursor: usize,
}

impl<N: DenseId> Walk<N> {
    fn new<G: RootedView<NodeId = N>>(graph: &G) -> Self {
        let mut walk = Self {
            blocks: Vec::new(),
            sources: Vec::new(),
            targets: Vec::new(),
            next: Vec::new(),
            returning: Vec::new(),
            waiting: Vec::new(),
        };
        let mut position = vec![NO_INDEX; graph.node_bound()];
        // Tarjan's offline nearest common ancestors: a finished position joins
        // the set of its parent, and `open` names, for each set, its one
        // position still on the stack. That is the nearest common ancestor of
        // any member and of the block being examined, which is below every
        // position on the stack. A position on the stack names itself.
        let mut finished = DisjointSet::new(0);
        let mut open: Vec<u32> = Vec::new();
        let mut arena: Vec<N> = Vec::new();
        let mut frames: Vec<Frame> = Vec::new();

        walk.enter(graph, graph.root(), &mut position, &mut arena, &mut frames);
        finished.push();
        open.push(0);
        while let Some(frame) = frames.last_mut() {
            let from = frame.position;
            let Some(&successor) = arena.get(frame.cursor) else {
                arena.truncate(frame.start);
                frames.pop();
                if let Some(parent) = frames.last() {
                    finished.union(slot(from), slot(parent.position));
                    open[finished.find(slot(from))] = parent.position;
                }
                continue;
            };
            frame.cursor += 1;
            match position[successor.index()] {
                NO_INDEX => {
                    let to = walk.enter(graph, successor, &mut position, &mut arena, &mut frames);
                    finished.push();
                    open.push(to);
                    walk.file(from, to, from);
                }
                to => walk.file(from, to, open[finished.find(slot(to))]),
            }
        }
        walk
    }

    /// Give `block` the next position, and start examining its successors.
    fn enter<G: RootedView<NodeId = N>>(
        &mut self,
        graph: &G,
        block: N,
        position: &mut [u32],
        arena: &mut Vec<N>,
        frames: &mut Vec<Frame>,
    ) -> u32 {
        let at = word(self.blocks.len());
        position[block.index()] = at;
        self.blocks.push(block);
        self.returning.push(NO_INDEX);
        self.waiting.push(NO_INDEX);
        let start = arena.len();
        arena.extend(graph.successors(block));
        frames.push(Frame {
            position: at,
            start,
            cursor: start,
        });
        at
    }

    /// File the edge `from -> to`, whose ends have `ancestor` as nearest common
    /// ancestor: as returning to `to` when that is `to` itself, and otherwise
    /// as waiting at `ancestor`.
    fn file(&mut self, from: u32, to: u32, ancestor: u32) {
        let edge = word(self.sources.len());
        self.sources.push(from);
        self.targets.push(to);
        let head = if ancestor == to {
            &mut self.returning[slot(to)]
        } else {
            &mut self.waiting[slot(ancestor)]
        };
        self.next.push(*head);
        *head = edge;
    }
}

/// The cycles of a walk, by position, innermost first.
struct Cycles {
    /// The header of each cycle.
    headers: Vec<u32>,
    /// The cycle directly around each one, [`NO_INDEX`] for an outermost one.
    parents: Vec<u32>,
    /// Where the latches of each cycle start in `latches`, with one more entry
    /// for the end.
    latch_starts: Vec<u32>,
    /// The sources of the edges returning to each header, one run per cycle.
    latches: Vec<u32>,
    /// The innermost cycle of each position, [`NO_INDEX`] for a position in none.
    innermost: Vec<u32>,
    /// The cycle each position heads, [`NO_INDEX`] for a position that heads none.
    headed: Vec<u32>,
    /// The outermost cycle each position enters other than as its header,
    /// [`NO_INDEX`] for a position that enters none.
    entered: Vec<u32>,
}

impl Cycles {
    /// Find the cycle of every position in reverse preorder, and collapse it
    /// into its header.
    fn collapse<N>(walk: &mut Walk<N>) -> Self {
        let count = walk.blocks.len();
        let mut cycles = Self {
            headers: Vec::new(),
            parents: Vec::new(),
            latch_starts: vec![0],
            latches: Vec::new(),
            innermost: vec![NO_INDEX; count],
            headed: vec![NO_INDEX; count],
            entered: vec![NO_INDEX; count],
        };
        // A set is a lone position or a finished cycle, and `header` names the
        // position it has collapsed into, by representative.
        let mut sets = DisjointSet::new(count);
        let mut header: Vec<u32> = (0..count).map(word).collect();
        let collapsed = |sets: &mut DisjointSet, header: &[u32], position: u32| {
            header[sets.find(slot(position))]
        };
        // The first edge handed to each collapsed position, and the header
        // whose walk last took each one in.
        let mut incoming = vec![NO_INDEX; count];
        let mut taken = vec![NO_INDEX; count];
        let mut body: Vec<u32> = Vec::new();
        for w in (0..count).rev() {
            let at = word(w);
            // The cycles collapsed so far are headed below `w`, so the target
            // of an edge waiting here enters the one it has been collapsed
            // into, and the visits of the ancestors of `w` find it there.
            let mut edge = walk.waiting[w];
            while edge != NO_INDEX {
                let following = walk.next[slot(edge)];
                let target = walk.targets[slot(edge)];
                let into = collapsed(&mut sets, &header, target);
                if into != target {
                    cycles.entered[slot(target)] = cycles.headed[slot(into)];
                }
                walk.next[slot(edge)] = incoming[slot(into)];
                incoming[slot(into)] = edge;
                edge = following;
            }

            let mut edge = walk.returning[w];
            if edge == NO_INDEX {
                continue;
            }
            body.clear();
            while edge != NO_INDEX {
                let source = walk.sources[slot(edge)];
                cycles.latches.push(source);
                let member = collapsed(&mut sets, &header, source);
                if member != at && taken[slot(member)] != at {
                    taken[slot(member)] = at;
                    body.push(member);
                }
                edge = walk.next[slot(edge)];
            }
            // Every edge handed to a member so far has its source below `w`,
            // since `w` is a common ancestor of its ends, so that source
            // reaches `w` inside the subtree and is in the cycle too.
            let mut cursor = 0;
            while let Some(&member) = body.get(cursor) {
                cursor += 1;
                let mut edge = incoming[slot(member)];
                while edge != NO_INDEX {
                    let source = collapsed(&mut sets, &header, walk.sources[slot(edge)]);
                    if source != at && taken[slot(source)] != at {
                        taken[slot(source)] = at;
                        body.push(source);
                    }
                    edge = walk.next[slot(edge)];
                }
            }

            let this = word(cycles.headers.len());
            cycles.headers.push(at);
            cycles.parents.push(NO_INDEX);
            cycles.latch_starts.push(word(cycles.latches.len()));
            cycles.innermost[w] = this;
            cycles.headed[w] = this;
            for &member in &body {
                match cycles.headed[slot(member)] {
                    NO_INDEX => cycles.innermost[slot(member)] = this,
                    inner => cycles.parents[slot(inner)] = this,
                }
                sets.union(slot(member), w);
            }
            header[sets.find(w)] = at;
        }
        cycles
    }

    /// The forest of these cycles, with loops numbered by ascending header.
    fn place<N: DenseId>(&self, walk: &Walk<N>, node_bound: usize) -> LoopForest<N> {
        let mut forest = LoopForest {
            loops: Vec::with_capacity(self.headers.len()),
            headers: Vec::with_capacity(self.headers.len()),
            latches: Vec::with_capacity(self.latches.len()),
            entries: Vec::new(),
            entered: Vec::new(),
            explicit: Vec::new(),
            innermost: Vec::new(),
            members: Vec::new(),
            member_starts: Vec::new(),
            node_bound,
        };
        if self.headers.is_empty() {
            return forest;
        }
        let mut ascending: Vec<usize> = (0..self.headers.len()).collect();
        ascending.sort_unstable_by_key(|&k| walk.blocks[slot(self.headers[k])]);
        let mut id = vec![NO_INDEX; self.headers.len()];
        for (index, &k) in ascending.iter().enumerate() {
            id[k] = word(index);
        }

        let entry_counts = self.entry_counts();
        let mut run: Vec<N> = Vec::new();
        for &k in &ascending {
            run.clear();
            run.extend(
                self.latches[slot(self.latch_starts[k])..slot(self.latch_starts[k + 1])]
                    .iter()
                    .map(|&latch| walk.blocks[slot(latch)]),
            );
            run.sort_unstable();
            run.dedup();
            let start = forest.latches.len();
            forest.latches.extend_from_slice(&run);
            forest.headers.push(walk.blocks[slot(self.headers[k])]);
            forest.loops.push(LoopRecord {
                latch_start: word(start),
                latch_end: word(forest.latches.len()),
                entry_start: 0,
                entry_end: 0,
                entry_count: entry_counts[k],
                explicit: NO_INDEX,
                family: 0,
                parent: NO_INDEX,
                depth: 0,
                enclosing: match self.parents[k] {
                    NO_INDEX => NO_INDEX,
                    parent => id[slot(parent)],
                },
                first: NO_INDEX,
                last: NO_INDEX,
                deepest: 0,
            });
        }
        forest.innermost = vec![NO_INDEX; node_bound];
        for (&block, &cycle) in walk.blocks.iter().zip(&self.innermost) {
            if cycle != NO_INDEX {
                forest.innermost[block.index()] = id[slot(cycle)];
            }
        }
        if entry_counts.iter().any(|&count| count > 1) {
            forest.entered = vec![NO_INDEX; node_bound];
            for (&block, &cycle) in walk.blocks.iter().zip(&self.entered) {
                if cycle != NO_INDEX {
                    forest.entered[block.index()] = id[slot(cycle)];
                }
            }
        }

        // The cycles were found innermost first, the order placing them needs.
        let order: Vec<usize> = id.iter().map(|&lp| slot(lp)).collect();
        forest.number_subtrees(&order);
        forest.group_members();
        forest.nest(&order);
        forest
    }

    /// The number of entries of every cycle, its header included.
    ///
    /// A position enters a run of cycles up the forest: from the innermost
    /// around it, or for a header from the cycle around the one it heads, to
    /// the outermost it enters. So a cycle's entries other than its header are
    /// the runs that start inside it, less those that also end strictly inside
    /// it, which sums over the subtrees count. A cycle comes before the one
    /// around it, so every sum is complete before it moves up.
    fn entry_counts(&self) -> Vec<u32> {
        let mut starts = vec![0_u32; self.headers.len()];
        let mut ends = vec![0_u32; self.headers.len()];
        for (position, &outermost) in self.entered.iter().enumerate() {
            if outermost != NO_INDEX {
                let first = match self.headed[position] {
                    NO_INDEX => self.innermost[position],
                    own => self.parents[slot(own)],
                };
                starts[slot(first)] += 1;
                ends[slot(outermost)] += 1;
            }
        }
        let mut ended_inside = vec![0_u32; self.headers.len()];
        for (k, &parent) in self.parents.iter().enumerate() {
            if parent != NO_INDEX {
                starts[slot(parent)] += starts[k];
                ended_inside[slot(parent)] += ended_inside[k] + ends[k];
            }
        }
        starts
            .iter()
            .zip(&ended_inside)
            .map(|(&started, &ended)| 1 + started - ended)
            .collect()
    }
}

#[cfg(test)]
mod tests;
