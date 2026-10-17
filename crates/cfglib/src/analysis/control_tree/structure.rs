//! CFG → control-tree structuring algorithm.
//!
//! Uses the dominator and post-dominator trees, dominance-proven natural
//! loops, and edge classifications to reconstruct structured control flow
//! from a [`Cfg`].

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::vec::Vec;

use super::node::{BlockSpan, ControlNode};
use super::report::{GotoDiagnostic, GotoReason, LiftReport};
use super::tree::{ControlTree, Tables};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::dominator::{DominatorTree, PostDominatorTree};
use crate::graph::structure::{NaturalLoop, detect_loops_tagged};
use crate::region::Region;
use roots::VirtualRoot;

mod branches;
mod labels;
mod loops;
mod roots;
mod try_catch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockFlowKind {
    LoopHeader,
    Conditional,
    Switch,
    BackEdge,
    Jump,
    Linear,
}

/// One entered loop, innermost last, resolving break/continue transfers.
struct LoopContext {
    /// The loop header block.
    header: BlockId,
    /// Block a `break` transfers to, when the loop has a recognized follow.
    follow: Option<BlockId>,
    /// Block a `continue` transfers to: the header, or a post-tested loop's
    /// condition latch.
    continue_target: BlockId,
    /// Set when a labeled break or continue named this loop.
    labeled: bool,
}

/// The working state of one structuring run.
struct Structurer<'cfg, I, E> {
    cfg: &'cfg Cfg<I, E>,
    /// Post-dominators over dense block indices, rooted at a virtual block
    /// that also precedes the region metadata blocks the entry cannot reach.
    pdom: PostDominatorTree<u32>,
    /// Whether the entry reaches each block, by dense index.
    reachable: Vec<bool>,
    /// The blocks the handlers own: entries, filter funclets, and the blocks
    /// of known extents. Sweep roots.
    handler_roots: Vec<BlockId>,
    /// Dominance-proven and explicitly tagged back-edge endpoint pairs.
    ///
    /// Machine frontends normally describe every native jump as
    /// [`EdgeKind::Jump`]. Keeping the structural result separately lets the
    /// structurer recover those loops without requiring frontends to mutate
    /// the lossless native edge classification.
    back_edges: BTreeSet<(BlockId, BlockId)>,
    /// Natural loops keyed by header, honoring tagged back edges.
    loops: BTreeMap<BlockId, NaturalLoop>,
    /// Region anchor (first protected block in reverse postorder) → the
    /// regions anchored there, outermost (largest protected set) first.
    anchors: BTreeMap<BlockId, Vec<&'cfg Region>>,
    visited: Vec<bool>,
    /// Blocks structured as if they held no instructions, by dense index.
    emptied: Vec<bool>,
    /// Blocks targeted by an emitted [`ControlNode::Goto`]; the label
    /// post-pass wraps their eventual emission in the matching label.
    goto_targets: BTreeSet<BlockId>,
    /// Blocks whose label was already emitted (swept sequences, labeled
    /// loops), so the post-pass does not label them twice.
    labeled_blocks: BTreeSet<BlockId>,
    /// Enclosing loops, innermost last.
    loop_stack: Vec<LoopContext>,
    /// Regions structured as [`ControlNode::TryCatch`].
    structured_regions: BTreeSet<u32>,
    /// The blocks no exception region names, as protected code, a handler
    /// entry, or declared handler extent. Region-claimed code never moves
    /// across structural boundaries, so code lifted out of its bound is
    /// bounded to these.
    unclaimed: Rc<BTreeSet<BlockId>>,
    report: LiftReport,
    tables: Tables,
}

impl ControlTree {
    /// Structure a [`Cfg`] into a control tree.
    ///
    /// Convenience wrapper over [`compute_with_report`](Self::compute_with_report)
    /// that discards the structural fidelity report.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, ControlNode, ControlTree, EdgeKind};
    ///
    /// // entry --true--> body --jump--> entry, entry --false--> exit
    /// let mut cfg = Cfg::<&str>::new();
    /// let body = cfg.new_block();
    /// let exit = cfg.new_block();
    /// cfg.block_mut(cfg.entry()).push("test");
    /// cfg.block_mut(body).push("work");
    /// cfg.block_mut(exit).push("done");
    /// cfg.add_edge(cfg.entry(), body, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(body, cfg.entry(), EdgeKind::Jump);
    ///
    /// let tree = ControlTree::compute(&cfg);
    /// let loops = tree
    ///     .preorder()
    ///     .filter(|&id| matches!(tree.node(id), ControlNode::Loop { .. }))
    ///     .count();
    /// assert_eq!(loops, 1);
    /// ```
    #[must_use]
    pub fn compute<I, E>(cfg: &Cfg<I, E>) -> Self {
        Self::compute_with_report(cfg).0
    }

    /// Structure a [`Cfg`] into a control tree and report exactly which
    /// parts degraded to unstructured flow.
    ///
    /// The structurer handles:
    /// - Structured flow: `IfThenElse`, `Loop` (classified pre-tested,
    ///   post-tested, or endless through [`LoopKind`](super::LoopKind)),
    ///   `Switch` with an explicit default arm and case edge identities
    /// - Loop transfers: `Break` / `Continue` derived from natural-loop
    ///   membership, labeled when they cross an enclosing loop
    /// - Exception regions: `TryCatch` when every handler has a complete
    ///   [`HandlerBody`](crate::HandlerBody) and one `try` can hold the
    ///   protected code (ordinary flow enters it at its first block only, and
    ///   none of it is already emitted or outside the enclosing bound);
    ///   multiple `Finally` handlers concatenate in declaration order
    /// - Unstructured flow: `Label` / `Goto` (for `Jump` edges no loop context
    ///   absorbs, transfers that leave a structural bound, and transfers into
    ///   already-emitted code)
    ///
    /// Other regions remain ordinary control flow: the structurer does not
    /// guess which blocks belong inside a handler, and does not show protected
    /// code outside the `try` that raises into its handler.
    ///
    /// A `try` body holds the protected code of its region. It may also hold
    /// code that no region names and only the protected body reaches, as a
    /// source `try { ...; return x; }` holds the `return` that an exception
    /// table leaves out, but never code that another region protects or
    /// handles: that code stays outside, behind a goto.
    /// **No owned code is ever dropped** (see below): a transfer refused at a
    /// structural boundary becomes an explicit [`ControlNode::Goto`], and
    /// every owned block the structured walk did not emit (refused targets,
    /// handler pads entered only through exception edges, dead pads and their
    /// tails, filter funclets) is appended at the end under its label. The returned
    /// [`LiftReport`] lists every emitted goto, swept block, and unstructured
    /// region so consumers can degrade per construct instead of guessing from
    /// the tree shape.
    ///
    /// # Merge points and exit-free regions
    ///
    /// A branch or switch rejoins at the immediate post-dominator of its
    /// block, read from a [`PostDominatorTree`]. That tree joins every exit
    /// to a virtual exit, and [`PostNode::Exit`](crate::PostNode::Exit) means the block has no merge
    /// point: its arms never rejoin, so each runs to its own end.
    ///
    /// A region that no path leaves, such as a server loop or a spin on a
    /// failed check, has no exit to post-dominate, so the tree cuts it: it
    /// joins the blocks that close the region's loops to the virtual exit.
    /// Facts about blocks outside such a region are unchanged, but a branch
    /// with an arm that never exits has no merge point even when its other
    /// arm leads to an exit:
    ///
    /// ```text
    /// entry --true--> spin (loops on itself)      if { loop { spin } } else { out }
    /// entry --false--> out (returns)
    /// ```
    ///
    /// A tree that left such regions out would make `out` the merge and give
    /// `if { loop { spin } }; out`, and a graph with no exit at all would need
    /// an arbitrary block as the exit. With the cut, the exiting arm is the
    /// `else` arm, and a transfer into an exit-free region stays explicit:
    /// when two arms enter the same exit-free loop, the structurer emits a
    /// [`ControlNode::Goto`] in the later one instead of ending it silently.
    ///
    /// # Arms that revisit earlier code
    ///
    /// A switch owns an arm's entry block only when nothing emitted or
    /// claimed it before. An arm that targets earlier code, such as the
    /// enclosing loop's header, a shared tail, or the block a sibling arm
    /// already owns (a default that shares a case's target), is a transfer: a
    /// `Continue`, a `Break`, or a [`ControlNode::Goto`]. The block is never
    /// emitted a second time, and a case that jumps back to its own loop header
    /// cannot recurse without end.
    ///
    /// # Code the entry does not reach
    ///
    /// A block is *owned* when the entry reaches it, or when a handler owns
    /// it: a handler entry, a filter funclet, or a block of a known handler
    /// extent. The tree holds the instructions of each owned block exactly
    /// once, whether or not an exception edge leads into the handler: a catch
    /// pad is dead when no call unwinds into it, yet its body is the source of
    /// a catch clause and still code. Code that only dead code reaches is
    /// dropped, and so is a protected block the entry cannot reach: such a
    /// block is unreachable source, not part of any handler. The exception
    /// is a goto — a transfer out of owned code to a block nothing owns finds
    /// its label, because the sweep emits that block and whatever it then
    /// names.
    ///
    /// - A handler body is lifted inside its handler: the walk from the pad,
    ///   then the rest of the extent that the entry cannot reach, in flow
    ///   order. The enclosing loop's bound holds the dead blocks that feed the
    ///   loop, and the handler's bound adds its own dead extent, so a tail that
    ///   cannot reach the loop stays in the handler. A pad of a region that
    ///   has not been structured yet stays with that region.
    /// - A reachable block of an extent that the walk did not reach waits for
    ///   the sweep, as it always has.
    /// - Branches in dead owned code merge as if the function began there too,
    ///   so a dead pad's `if` rejoins like any other; the post-dominator tree
    ///   is rooted at a virtual block that precedes the entry and every block
    ///   the entry misses. That changes no fact about a block the entry
    ///   reaches, and only blocks the entry reaches can pick a loop's follow.
    ///
    /// A region is well formed when its handlers' entries and extents are not
    /// among its protected blocks and ordinary flow enters them only from
    /// their own extent. The tree of a graph that breaks this may hold a block
    /// in two places, since the block is both a handler's and a loop's.
    ///
    /// # Stack depth
    ///
    /// Structuring recurses once per level of nested construct, and every link
    /// of an `else if` chain is one more level. The call stack therefore bounds
    /// the nesting depth of the input (a chain of a thousand links needs more
    /// than a default test thread's stack in a debug build). The pseudocode
    /// printer and [`compute_predicated`](Self::compute_predicated) recurse the
    /// same way; [`preorder`](Self::preorder), [`walk`](Self::walk), and
    /// [`for_each_instruction`](Self::for_each_instruction) do not.
    ///
    /// # Panics
    ///
    /// Panics (debug) when a [`HandlerBody::Known`](crate::HandlerBody::Known)
    /// omits its own handler entry — a frontend bug; release builds treat such
    /// a region as unstructured flow.
    #[must_use]
    pub fn compute_with_report<I, E>(cfg: &Cfg<I, E>) -> (Self, LiftReport) {
        Structurer::new(cfg, &[]).run()
    }

    /// Structure a [`Cfg`] as if the blocks of `empty` held no instructions,
    /// and report which parts degraded to unstructured flow.
    ///
    /// The structurer reads only how many instructions a block has, so the
    /// result is the tree [`compute_with_report`](Self::compute_with_report)
    /// gives for a copy of `cfg` whose `empty` blocks were cleared: an emptied
    /// block has no [`ControlNode::Block`] or [`ControlNode::Return`] leaf, its
    /// condition spans are empty, and an emptied block with a single
    /// [`EdgeKind::Jump`](crate::EdgeKind) successor is a trampoline that a
    /// `break` or `continue` routes through. `cfg` is not copied or changed.
    ///
    /// A lifter uses this for blocks whose only instruction is a pure transfer
    /// that the structure already expresses, such as the jump a source `break`
    /// compiles to.
    ///
    /// # Panics
    ///
    /// Panics when a block of `empty` lies beyond `cfg`'s
    /// [`block_bound`](Cfg::block_bound), and under the conditions of
    /// [`compute_with_report`](Self::compute_with_report).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, ControlTree, EdgeKind};
    ///
    /// // entry --true--> hop --jump--> exit; entry --false--> exit
    /// let mut cfg = Cfg::<&str>::new();
    /// let hop = cfg.new_block();
    /// let exit = cfg.new_block();
    /// cfg.block_mut(cfg.entry()).push("test");
    /// cfg.block_mut(hop).push("jmp");
    /// cfg.block_mut(exit).push("done");
    /// cfg.add_edge(cfg.entry(), hop, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(hop, exit, EdgeKind::Jump);
    ///
    /// let (tree, _) = ControlTree::compute_with_report_treating_empty(&cfg, &[hop]);
    /// let text = tree
    ///     .to_pseudocode(|sink, point| {
    ///         sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
    ///     })
    ///     .unwrap();
    /// assert!(!text.contains("jmp"));
    /// ```
    #[must_use]
    pub fn compute_with_report_treating_empty<I, E>(
        cfg: &Cfg<I, E>,
        empty: &[BlockId],
    ) -> (Self, LiftReport) {
        Structurer::new(cfg, empty).run()
    }
}

impl<'cfg, I, E> Structurer<'cfg, I, E> {
    fn new(cfg: &'cfg Cfg<I, E>, empty: &[BlockId]) -> Self {
        let mut emptied = alloc::vec![false; cfg.block_bound()];
        for block in empty {
            emptied[block.index()] = true;
        }
        let dom = DominatorTree::compute(cfg);
        // A latch the entry cannot reach never runs, so it is dead code that
        // feeds a live header, not a back edge of the live program: a block
        // that returns would otherwise sit in an endless loop. A dead loop (a
        // header the entry cannot reach either) keeps its latches.
        let loops: BTreeMap<BlockId, NaturalLoop> = detect_loops_tagged(cfg, &dom)
            .into_iter()
            .filter_map(|mut natural| {
                if dom.is_reachable(natural.header) {
                    natural.latches.retain(|&latch| dom.is_reachable(latch));
                }
                (!natural.latches.is_empty()).then_some((natural.header, natural))
            })
            .collect();
        let back_edges = loops
            .values()
            .flat_map(|natural| natural.latches.iter().map(|&latch| (latch, natural.header)))
            .collect();
        let mut reachable = alloc::vec![false; cfg.block_bound()];
        for block in cfg.block_ids() {
            reachable[block.index()] = dom.is_reachable(block);
        }
        // The post-dominator tree holds only the blocks its root reaches. The
        // structurer emits code the entry cannot reach — a handler pad with no
        // exception edge, a filter funclet — and every such block needs merge
        // points too, so the root also precedes each block the entry misses.
        let unreachable = cfg.block_ids().filter(|block| !reachable[block.index()]);
        Self {
            cfg,
            pdom: PostDominatorTree::compute(&VirtualRoot::new(cfg, unreachable)),
            handler_roots: try_catch::handler_roots(cfg, |block| {
                !emptied[block.index()] && !cfg.block(block).instructions().is_empty()
            }),
            reachable,
            back_edges,
            loops,
            anchors: BTreeMap::new(),
            visited: alloc::vec![false; cfg.block_bound()],
            emptied,
            goto_targets: BTreeSet::new(),
            labeled_blocks: BTreeSet::new(),
            loop_stack: Vec::new(),
            structured_regions: BTreeSet::new(),
            unclaimed: Rc::new(unclaimed_blocks(cfg)),
            report: LiftReport::default(),
            tables: Tables::new(),
        }
    }

    fn run(mut self) -> (ControlTree, LiftReport) {
        let cfg = self.cfg;
        let order = cfg.reverse_postorder();
        self.anchors = try_catch::region_anchors(cfg, &order);
        let mut body = self.lift_region(cfg.entry(), None, None);

        // Completeness sweep: emit every block the structured walk refused or
        // never reached, each under its label so the boundary `Goto`s and
        // exception edges that lead there stay resolvable. The entry's blocks
        // come first, then the handlers' own blocks — a pad with no exception
        // edge or a funclet carries code, though no edge from the entry leads
        // to it.
        let sweep_order = self.sweep_order(&order);
        while let Some(&pending) = sweep_order.iter().find(|block| !self.is_visited(**block)) {
            self.sweep(pending, &mut body);
        }
        // A goto always finds its label: the target of one that left a handler
        // for code only that handler reaches is swept too, and so is whatever
        // the sweep then names.
        while let Some(target) = self
            .goto_targets
            .iter()
            .copied()
            .find(|&target| !self.is_visited(target))
        {
            self.sweep(target, &mut body);
        }

        // Exact labeling: wrap the emission of every goto target that has no
        // label yet; whatever cannot be anchored is a dangling goto worth
        // reporting.
        let root_body = self.tables.seal(body);
        let mut pending_labels: BTreeSet<BlockId> = self
            .goto_targets
            .difference(&self.labeled_blocks)
            .copied()
            .collect();
        self.apply_labels(root_body, &mut pending_labels);
        self.report.unresolved_labels.extend(pending_labels);
        for region in cfg.regions() {
            // Inert tombstones (removed regions) carry nothing to structure.
            if region.protected_blocks.is_empty() && region.handlers.is_empty() {
                continue;
            }
            if !self.structured_regions.contains(&region.id.0) {
                self.report.unstructured_regions.push(region.id);
            }
        }

        // A body of one statement is that statement.
        let root = match self.tables.body(root_body) {
            &[only] => only,
            _ => self.tables.push_node(ControlNode::Sequence(root_body)),
        };
        (self.tables.finish(root), self.report)
    }

    /// Emit the code at `pending`, which nothing structured reached, under
    /// its label at the end of `body`.
    fn sweep(&mut self, pending: BlockId, body: &mut Vec<ControlNode>) {
        let pending_body = self.lift_region(pending, None, None);
        if pending_body.is_empty() {
            return;
        }
        self.report.swept_blocks.push(pending);
        if self.labeled_blocks.insert(pending) {
            let swept = self.tables.seal(pending_body);
            body.push(ControlNode::Label {
                block: pending,
                body: swept,
            });
        } else {
            // A labeled loop headed by `pending` already carries the label: a
            // second one would name the block twice.
            body.extend(pending_body);
        }
    }

    /// The blocks the completeness sweep walks, in order: reverse postorder
    /// from the entry, then the handlers' own blocks the entry misses.
    fn sweep_order(&self, order: &[BlockId]) -> Vec<BlockId> {
        let mut listed = alloc::vec![false; self.cfg.block_bound()];
        let mut sweep = order.to_vec();
        for block in order {
            listed[block.index()] = true;
        }
        for &root in &self.handler_roots {
            if !core::mem::replace(&mut listed[root.index()], true) {
                sweep.push(root);
            }
        }
        sweep
    }

    /// Whether an exception region names `block`.
    fn is_claimed(&self, block: BlockId) -> bool {
        !self.unclaimed.contains(&block)
    }

    fn is_visited(&self, block: BlockId) -> bool {
        self.visited[block.index()]
    }

    fn visit(&mut self, block: BlockId) {
        self.visited[block.index()] = true;
    }

    fn is_back_edge(&self, edge: EdgeId) -> bool {
        let edge = self.cfg.edge(edge);
        self.back_edges.contains(&(edge.source(), edge.target()))
    }

    /// The target of the back edge that leaves `block` by sequential flow.
    ///
    /// An exception edge is not sequential flow, so a handler entry that
    /// dominates its protected block does not make that block a latch: the
    /// block keeps its ordinary successors and the handler arm carries the
    /// transfer to the header.
    fn flow_back_edge_target(&self, block: BlockId) -> Option<BlockId> {
        let cfg = self.cfg;
        cfg.outgoing(block)
            .map(|edge| cfg.edge(edge))
            .find(|edge| {
                !edge.kind().is_exceptional()
                    && self.back_edges.contains(&(edge.source(), edge.target()))
            })
            .map(|edge| edge.target())
    }

    fn classify_block(&self, block: BlockId) -> BlockFlowKind {
        let cfg = self.cfg;
        let successors: Vec<EdgeId> = cfg.outgoing(block).collect();
        if cfg.incoming(block).any(|edge| self.is_back_edge(edge)) {
            BlockFlowKind::LoopHeader
        } else if let Some(branch) = branching(cfg, &successors) {
            branch
        } else if self.flow_back_edge_target(block).is_some() {
            BlockFlowKind::BackEdge
        } else if has_edge_kind(cfg, &successors, EdgeKind::Jump) {
            BlockFlowKind::Jump
        } else {
            BlockFlowKind::Linear
        }
    }

    /// How many instructions the structure sees in `block`: none for a block
    /// the caller emptied.
    fn instruction_count(&self, block: BlockId) -> usize {
        if self.emptied[block.index()] {
            0
        } else {
            self.cfg.block(block).instructions().len()
        }
    }

    /// The span of every instruction of `block`.
    fn whole_span(&self, block: BlockId) -> BlockSpan {
        BlockSpan::whole(block, self.instruction_count(block))
    }

    /// Emit `block`'s instructions as a leaf; a block without instructions
    /// has nothing to emit.
    fn push_block(&self, result: &mut Vec<ControlNode>, block: BlockId) {
        let span = self.whole_span(block);
        if !span.is_empty() {
            result.push(ControlNode::Block(span));
        }
    }

    /// Recursively lift a region starting at `head`.
    ///
    /// The walk ends silently at `stop` (a legitimate structural convergence
    /// point — an arm's merge, a post-tested loop's latch); every other
    /// transfer that cannot proceed is resolved against the loop stack or made
    /// explicit as a goto.
    fn lift_region(
        &mut self,
        head: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
        stop: Option<BlockId>,
    ) -> Vec<ControlNode> {
        let cfg = self.cfg;
        let mut result = Vec::new();
        let mut current = Some(head);

        while let Some(block) = current {
            if Some(block) == stop {
                break;
            }
            if let Some(node) = self.resolve_loop_transfer(block) {
                result.push(node);
                break;
            }
            if self.is_visited(block) {
                // Convergence is only silent at `stop`; any other transfer
                // into already-emitted code is a real control transfer.
                self.push_goto(&mut result, block, GotoReason::RevisitedTarget);
                break;
            }
            if !block_is_allowed(allowed_blocks, block) {
                // Exclusively owned out-of-bound code — a single predecessor,
                // so this transfer is its only entry, and no exception region
                // claims it — inlines here instead of degrading to a goto
                // (javac hoists `break label` bodies out of line like this).
                // The inlined walk stays within unclaimed code, so it never
                // pulls another region's blocks into this construct.
                // Anything else is recorded, never dropped: the jump is
                // explicit and the completeness sweep emits the target under
                // the matching label.
                if cfg.incoming(block).count() == 1 && !self.is_claimed(block) {
                    let bound = Rc::clone(&self.unclaimed);
                    result.extend(self.lift_region(block, Some(&bound), stop));
                } else {
                    self.push_goto(&mut result, block, GotoReason::BoundaryEscape);
                }
                break;
            }

            self.visit(block);
            current = self.lift_entered(block, allowed_blocks, stop, &mut result);
        }

        result
    }

    /// Emit an already-visited block: the try region it anchors, or else its
    /// own classified form. Returns the block flow continues at (if any).
    ///
    /// [`lift_region`](Self::lift_region)'s walk and a switch arm, whose entry
    /// the switch visits ahead of the walk, both enter blocks here, so a
    /// region anchored at an arm entry is structured like any other.
    pub(super) fn lift_entered(
        &mut self,
        block: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
        stop: Option<BlockId>,
        result: &mut Vec<ControlNode>,
    ) -> Option<BlockId> {
        if let Some((node, continuation)) = self.lift_try_catch(block, allowed_blocks) {
            result.push(node);
            // Resume at the region's own continuation, not the anchor's merge:
            // they differ whenever the try exits through an explicit leave.
            return self.resume_after(&node, continuation);
        }
        self.lift_block(block, allowed_blocks, stop, result)
    }

    /// Classify one already-visited block, emit its lifted form, and return
    /// the block sequential flow continues at (if any).
    ///
    /// This is the single classification path: [`lift_entered`](Self::lift_entered)
    /// and the try-body anchor both run it, so a region anchored at a branch,
    /// dispatch, or loop header keeps its structure inside the enclosing
    /// construct.
    fn lift_block(
        &mut self,
        block: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
        stop: Option<BlockId>,
        result: &mut Vec<ControlNode>,
    ) -> Option<BlockId> {
        let cfg = self.cfg;
        let outgoing: Vec<EdgeId> = cfg.outgoing(block).collect();
        let flow = self.classify_block(block);

        match flow {
            BlockFlowKind::LoopHeader => {
                let (node, follow) = self.lift_loop(block, allowed_blocks);
                result.push(node);
                return follow;
            }
            BlockFlowKind::Conditional => {
                let node = self.lift_conditional(block, allowed_blocks);
                result.push(node);
                let merge = self.effective_merge(block, allowed_blocks);
                return self.resume_after(&node, merge);
            }
            BlockFlowKind::Switch => {
                let node = self.lift_switch(block, allowed_blocks);
                result.push(node);
                let merge = self.effective_merge(block, allowed_blocks);
                return self.resume_after(&node, merge);
            }
            BlockFlowKind::BackEdge => {
                self.push_block(result, block);
                if let Some(target) = self.flow_back_edge_target(block) {
                    if let Some(node) = self.resolve_loop_transfer(target) {
                        result.push(node);
                    } else {
                        self.push_goto(result, target, GotoReason::RevisitedTarget);
                    }
                }
                return None;
            }
            BlockFlowKind::Jump => {
                return self.lift_jump(block, &outgoing, stop, result);
            }
            BlockFlowKind::Linear => {}
        }

        if outgoing.is_empty() {
            let span = self.whole_span(block);
            if !span.is_empty() {
                result.push(ControlNode::Return(span));
            }
            return None;
        }

        self.push_block(result, block);

        // Advance along the single sequential successor; exception edges are
        // not sequential flow, so they never block the advance. The walk in
        // [`lift_region`](Self::lift_region) resolves the target against the
        // stop block, the loop stack, and the enclosing bound.
        let flow_succs = flow_successors(cfg, block);
        if let &[only] = flow_succs.as_slice() {
            return Some(only);
        }
        None
    }

    /// Emit a block whose flow leaves through explicit jumps, and return the
    /// block sequential flow continues at (if any).
    fn lift_jump(
        &mut self,
        block: BlockId,
        outgoing: &[EdgeId],
        stop: Option<BlockId>,
        result: &mut Vec<ControlNode>,
    ) -> Option<BlockId> {
        let cfg = self.cfg;
        self.push_block(result, block);
        for &eid in outgoing {
            let edge = cfg.edge(eid);
            if edge.kind() != EdgeKind::Jump {
                continue;
            }
            let target = edge.target();
            if Some(target) == stop {
                // The jump lands exactly on the enclosing construct's
                // convergence point (an arm skipping its sibling to reach the
                // join): structurally the arm simply ends.
                continue;
            }
            if let Some(node) = self.resolve_loop_transfer(target) {
                result.push(node);
            } else if !self.is_visited(target)
                && cfg.incoming(target).count() == 1
                && !self.is_claimed(target)
            {
                // Exclusively owned jump target — this explicit jump is its
                // only entry and no exception region claims it (javac jumps
                // over handler code to reach a join): sequential flow simply
                // continues there.
                return Some(target);
            } else {
                self.push_goto(result, target, GotoReason::ExplicitJump);
            }
        }
        None
    }

    /// The ultimate target of a chain of pure jump trampolines: an unemitted,
    /// instruction-free block with a single unconditional successor forwards
    /// a resolution unchanged. The HLIL lift empties dialect-declared pure
    /// transfers in its working view before structuring (javac's `break`
    /// routes through such blocks); frontends structuring a raw [`Cfg`] whose
    /// jump encodings occupy an instruction clear them the same way.
    fn through_trampolines(&self, target: BlockId) -> (BlockId, Vec<BlockId>) {
        let cfg = self.cfg;
        let mut hops = Vec::new();
        let mut current = target;
        for _ in 0..8 {
            if self.is_visited(current) || self.instruction_count(current) != 0 {
                break;
            }
            let mut normal = cfg
                .outgoing(current)
                .map(|edge| cfg.edge(edge))
                .filter(|edge| !edge.kind().is_exceptional());
            let (Some(edge), None) = (normal.next(), normal.next()) else {
                break;
            };
            if edge.kind() != EdgeKind::Jump {
                break;
            }
            hops.push(current);
            current = edge.target();
        }
        (current, hops)
    }

    /// Resolve a control transfer against the enclosing loops: `continue`
    /// when it reaches a loop's continue point, `break` when it reaches a
    /// loop's follow. Inner loops win; resolving against an outer loop labels
    /// it.
    fn resolve_loop_transfer(&mut self, target: BlockId) -> Option<ControlNode> {
        let (target, hops) = self.through_trampolines(target);
        let position = self.loop_stack.iter().rposition(|context| {
            context.continue_target == target || context.follow == Some(target)
        })?;
        // The consumed trampolines carry no content; the sweep must not
        // resurrect them as labeled residue.
        for hop in hops {
            self.visit(hop);
        }
        let innermost = position + 1 == self.loop_stack.len();
        let label = if innermost {
            None
        } else {
            self.loop_stack[position].labeled = true;
            Some(self.loop_stack[position].header)
        };
        let context = &self.loop_stack[position];
        Some(if context.continue_target == target {
            ControlNode::Continue { label }
        } else {
            ControlNode::Break { label }
        })
    }

    fn push_goto(&mut self, result: &mut Vec<ControlNode>, target: BlockId, reason: GotoReason) {
        self.goto_targets.insert(target);
        self.report.gotos.push(GotoDiagnostic { target, reason });
        result.push(ControlNode::Goto { target });
    }
}

fn has_edge_kind<I, E>(cfg: &Cfg<I, E>, edges: &[EdgeId], kind: EdgeKind) -> bool {
    edges.iter().any(|&edge| cfg.edge(edge).kind() == kind)
}

/// How a block with outgoing `edges` branches: [`BlockFlowKind::Conditional`]
/// for a true and a false edge, [`BlockFlowKind::Switch`] for a case edge, and
/// `None` for a block that does not branch.
fn branching<I, E>(cfg: &Cfg<I, E>, edges: &[EdgeId]) -> Option<BlockFlowKind> {
    if has_edge_kind(cfg, edges, EdgeKind::ConditionalTrue)
        && has_edge_kind(cfg, edges, EdgeKind::ConditionalFalse)
    {
        Some(BlockFlowKind::Conditional)
    } else if has_edge_kind(cfg, edges, EdgeKind::SwitchCase) {
        Some(BlockFlowKind::Switch)
    } else {
        None
    }
}

/// Successors reached by sequential control flow, excluding exception edges.
fn flow_successors<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> Vec<BlockId> {
    cfg.outgoing(block)
        .filter(|&edge| !cfg.edge(edge).kind().is_exceptional())
        .map(|edge| cfg.edge(edge).target())
        .collect()
}

fn block_is_allowed(allowed_blocks: Option<&BTreeSet<BlockId>>, block: BlockId) -> bool {
    allowed_blocks.is_none_or(|blocks| blocks.contains(&block))
}

/// The blocks no exception region names: neither protected, nor a handler
/// entry, nor part of a declared handler extent.
fn unclaimed_blocks<I, E>(cfg: &Cfg<I, E>) -> BTreeSet<BlockId> {
    cfg.block_ids()
        .filter(|&block| {
            !cfg.regions().iter().any(|region| {
                region.protected_blocks.contains(&block)
                    || region.handlers.iter().any(|handler| handler.claims(block))
            })
        })
        .collect()
}
