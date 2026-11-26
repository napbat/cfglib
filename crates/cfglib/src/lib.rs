//! Generic graph and dataflow framework for code intelligence and program analysis.
//!
//! [`Graph`] is the one store: a compressed base plus an appendable delta,
//! holding arbitrary node and edge payloads for value-flow, symbol,
//! type-relation, import, call, and grammar graphs, with constant-time
//! addition, bit-cheap removal, identity-preserving endpoint redirection, and
//! an explicit [`compact`](Graph::compact) that numbers the surviving nodes
//! in insertion or a locality-friendly [`NodeOrder`] and reports its
//! [`Renumbering`].
//! Identities are one type, [`Id<T>`](Id), tagged per entity kind.
//! [`Fanout`], [`DenseFanout`], and [`SortedMap`] are that base's shape
//! without the edges: built-once tables from sparse keys, from a dense
//! `0..bound` key space, and from sparse keys to one value, so a reverse
//! index or a per-scope row table costs a few allocations rather than one
//! per key.
//! Algorithms consume [`GraphView`] / [`RootedView`], while [`NodeView`],
//! [`EdgeView`], and [`FilteredEdges`] retain payloads and edge identity
//! without rebuilding, so consumer-owned graph stores participate without
//! migrating their data.
//! [`Rooted`], [`Reversed`], [`Induced`], and [`VirtualRoot`] reshape a view
//! without copying it. [`depth_first_events`] exposes traversal-tree
//! structure for dense graphs; [`open_breadth_first_events`] and
//! [`open_depth_first_events`] provide the corresponding discovery streams
//! for lazily generated node spaces.
//! [`graph::scope::ScopeGraph`] stores language-defined scopes, labeled
//! reachability, relation-tagged declarations, and references.
//! [`graph::scope::ScopeGraphQuery`] supplies
//! the path automaton, matching, label/data specificity, or a full path order;
//! the same resolver handles stored references and ephemeral completion
//! requests and builds forward/reverse binding indexes.
//! [`graph::scope::ScopeEnvironmentQuery`] is the path-free form whose
//! per-scope environments a [`ScopeEnvironmentCache`] computes once and
//! shares across references.
//! [`graph::stack::StackGraph`] provides the complementary file-incremental
//! model with the standard root, scope, symbol push/pop, scoped-symbol,
//! drop-scopes, and jump-to-scope nodes. It supports concrete direct paths,
//! edge-precedence shadowing, serializable per-file partial-path databases, and
//! query-time stitching across independently built file partitions.
//! [`Cfg<I, E>`] adds basic-block, control-flow, and caller-owned edge metadata
//! when the graph really is a program CFG, and every
//! instruction-adjacent axis — variables, constants, operators, effects,
//! branch targets, callees — is consumer-typed rather than imposed by the
//! library.
//! [`ir`] holds the IR traits and vocabulary.
//! [`ir::rtl`] is the register-transfer contract: a write-only
//! [`ir::rtl::InstructionBuilder`] through which instruction semantics
//! describe one atomic state update, and read-only views over the sealed
//! records. [`ir::mlil`] is the read-only view contract of a flat static
//! single assignment function with block parameters; value analyses over it
//! belong to the consumer. [`ir::hlil`] is the structured contract: a
//! read-only [`ir::hlil::FunctionView`] of statement and expression trees and
//! a write-only [`ir::hlil::FunctionBuilder`] that the consumer implements.
//! Every IR storage, builder implementation, editing machinery, and
//! algorithm over the contents of an instruction belongs to the consumer.
//! [`effects`] is the one exception: [`Effect`] states what code reads,
//! writes, accesses in memory, and raises, and [`Effect::compute`] computes
//! it from register-transfer statements with one set of rules for every
//! level and every consumer. [`Captured`] holds the statements of one
//! instruction, and [`StatementTree`] turns them into expression trees for a
//! readable view.
//! [`ControlTree::compute`] structures a [`Cfg`] into `if / else`, loops,
//! `switch`, `try / catch`, and `goto` form. The tree is a fact about the graph
//! in the way [`DominatorTree`] is: a flat arena of nodes that name blocks,
//! edges, regions, and instruction spans, holding no instruction of its own.
//! Output is three formats over one set of writers. [`write_dot`] draws any
//! node- and edge-bearing view through a [`DotStyle`] — a node-label hook, an
//! edge-attribute hook ([`DotEdgeAttributes`]), a graph name, an identifier
//! prefix, and a [`DotRankDir`] — so [`Cfg`] and [`Graph`] are two styles
//! rather than two writers. [`write_text`] prints the same view one fact per
//! line, and [`parse_text`] reads it back into a plain [`Graph`], which is
//! what makes the form comparable: write, parse, write again is a fixed
//! point. [`Cfg::to_listing`] is the control-flow form — block headers
//! with indented instructions, edges named by [`EdgeKind`], and exception
//! regions — and [`parse_cfg_listing`] reads that back with a consumer-supplied
//! instruction parser. Every printer shares [`IndentedWriter`], so
//! indentation is decided once and a label is written straight into the sink,
//! and [`render`] turns any printer into a `String`.
//! [`PassPipeline`] composes named, ordered, fallible transformations over a
//! consumer-owned target, retaining a change report and failed-pass identity
//! without imposing a dialect policy.
//!
//! # Quick start
//!
//! Direct structural construction is the primary front door — no trait is
//! required to build, verify, analyze, or render a CFG:
//!
//! ```rust
//! use cfglib::{Cfg, DominatorTree, EdgeKind, verify};
//!
//! // A source frontend lowers its syntax tree straight into blocks.
//! let mut cfg = Cfg::<&'static str>::new();
//! let then_block = cfg.new_block();
//! let merge = cfg.new_block();
//! cfg.block_mut(cfg.entry()).push("if x > 0");
//! cfg.block_mut(then_block).push("y = x");
//! cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
//! cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
//! cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
//!
//! assert!(verify(&cfg).is_ok());
//! let dominators = DominatorTree::compute(&cfg);
//! assert!(dominators.dominates(cfg.entry(), merge));
//! ```
//!
//! Frontends with a flat, structured instruction stream (shader bytecode,
//! structured ISAs) can instead implement [`FlowControl`] and use
//! [`CfgBuilder::build`]; [`resolve_jump_edges`] wires the targets that
//! their [`Flow`]s name afterwards.
//!
//! # Extension contracts
//!
//! [`Graph`] owns arbitrary graph storage without requiring a consumer
//! trait. Existing graph stores implement [`DenseId`] and [`GraphView`]
//! (plus [`NodeView`] for graph-owned payloads, [`EdgeView`] for
//! edge-sensitive algorithms, and [`RootedView`] or the [`Rooted`] adapter
//! for entry-requiring algorithms) to reuse the generic algorithms. A view
//! reports a **bound** that sizes dense side tables and separately yields its
//! **live** identities, so a store with removed entities in it is neither
//! unsound to analyze nor obliged to compact first.
//! Instruction types implement progressively richer traits only when they
//! need CFG or dataflow facilities — every associated type below is the
//! consumer's own:
//!
//! ```text
//! Graph<N, E>               (owned arbitrary graph; no adapter trait)
//! GraphView                 (existing consumer-owned graph storage)
//!   ├─ NodeView             (graph-owned node payloads)
//!   ├─ EdgeView             (stable edge identity, endpoints, data)
//!   └─ RootedView           (adds a distinguished entry node; `Rooted` adapts)
//! CfgView                   (RootedView + EdgeView + block instruction slices;
//!                            `Cfg` implements it, every instruction-level
//!                            analysis reads it)
//!
//! FlowControl               (required by CfgBuilder and call graphs —
//!                            structured markers or a Flow; Target, CaseKey)
//!
//! InstrInfo<Variable = V>   (optional — native IR variables for dataflow)
//!   └─ MemoryEventInfo      (optional — memory data flow + fences, Location + Fence)
//!
//! DisplayInstr              (optional — rendering only: DOT, text)
//! ```
//!
//! [`MemoryEventInfo`] is the sole instruction-side memory contract. Memory
//! SSA and memory-carried value flow over it live in the `il` crate.
//!
//! Additionally, [`NodeProblem`] is the trait for pluggable dataflow analyses
//! with one fact per node of any graph view (per-block problems of a CFG
//! included), run by [`solve_node_problem`]; [`EdgeProblem`] is its
//! edge-sensitive counterpart, run by [`solve_edge_problem`]; each has a
//! fallible form ([`TryNodeProblem`], [`TryEdgeProblem`]) that every
//! infallible problem implements; and [`Emitter`] is the trait for
//! linearization output. A [`Solve`] names the [`Seeds`] a solve starts from
//! and its [`SolveConfig`], which selects the [`IterationOrder`]; in
//! [`IterationOrder::WeakTopological`] the solvers follow a
//! [`WeakTopologicalOrder`] and widen at its heads, so infinite-height
//! lattices converge. [`stream_node_problem`] and [`stream_edge_problem`]
//! run that solve and hand each node's facts out once they are final, instead
//! of keeping every fact until the end. [`resume_node_problem`] and
//! [`resume_edge_problem`] re-solve after an edit from the facts of the solve
//! before it, and [`OrderedFixpoint`] keeps the reads of the ordered slot
//! driver between runs.
//!
//! # Contracts
//!
//! - Blocks may be empty, may lack explicit terminator instructions, and
//!   unreachable blocks are legal (dead code after a return/goto).
//! - [`Cfg::blocks`] iterates live blocks in allocation order and
//!   [`Cfg::edges`] live edges in insertion order; both orders are stable and
//!   part of the API. A removed block or edge is skipped, keeps its slot, and
//!   stays readable through [`Cfg::block`] until [`Cfg::compact`] renumbers
//!   the survivors.
//! - [`ProgramPoint`] instruction indices are positions, not identities:
//!   [`Cfg::split_block`] and instruction edits invalidate them. Persist
//!   consumer-keyed results (e.g. by syntax-node id), not program points.
//! - Analyses size dense side tables by [`Cfg::block_bound`] /
//!   [`GraphView::node_bound`] and iterate [`Cfg::block_ids`] /
//!   [`GraphView::node_ids`]; the two coincide until something is removed.
//!   Consumer block anchors (source ranges, syntax nodes) belong in those
//!   side tables or in the instruction payload itself.
//! - A transform reports what it did as a [`Rewrite`] (sparse: only the
//!   identities it touched) and a compaction as a [`Renumbering`] (total over
//!   the old identity space). Both compose with `compose(&next)`, and
//!   `rewrite.compose(&Rewrite::from(&renumbering))` carries the first
//!   across the second.

#![no_std]

// Golden-file comparison reads the rendered output back from disk, which the
// crate itself never does.
#[cfg(test)]
extern crate std;

/// A count as a ratio operand. Counts of blocks, edges, and instructions stay
/// far below 2^53, where `f64` is exact.
#[expect(
    clippy::cast_precision_loss,
    reason = "a count of in-memory items fits f64's 53-bit mantissa"
)]
pub(crate) fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

pub mod analysis;
pub mod block;
pub mod builder;
pub mod cfg;
pub mod dataflow;
pub mod display;
pub mod edge;
pub mod effects;
pub mod exception;
pub mod flow;
pub mod graph;
mod identity;
pub mod ir;
pub mod memory;
pub mod region;
pub mod rewrite;
pub mod transform;
pub mod union_find;

#[cfg(test)]
pub(crate) mod test_util;

pub use analysis::control_tree::{
    BlockSpan, CatchHandler, ChildBodies, ControlBody, ControlDescent, ControlNode, ControlNodeId,
    ControlRun, ControlTree, ControlVisitor, GotoDiagnostic, GotoReason, LoopKind,
    PathDecomposition, Preorder, SplDecomposition, SplError, SplNode, SplNodeId, StructuringReport,
    SwitchCase,
};
pub use analysis::metrics::{CfgMetrics, GraphMetrics};
pub use analysis::pattern::{Pattern, detect_patterns, empty_trampolines};
pub use block::{BasicBlock, BlockId, BlockTag};
pub use builder::address::{
    AddressBuildError, AddressCfgOptions, AddressEdgeInfo, AddressGraph, AddressHandler,
    AddressHandlerKind, AddressInstruction, AddressSpace, CallPolicy, GivenEdge, GivenEdges,
    HopEnd, UnwindEnd, UnwindHop, UnwindPath, UnwindRoute, UnwindStep, build_address_cfg,
};
pub use builder::{BuildError, CfgBuilder, JumpResolution, resolve_jump_edges};
pub use cfg::{
    Cfg, CfgEdge, SemanticValidator, SplitPointError, control_flow_edge_attributes,
    parse_cfg_listing, verify, verify_semantics,
};
pub use dataflow::bits::DenseBits;
pub use dataflow::cfg_view::CfgView;
pub use dataflow::edge_fixpoint::{EdgeFacts, EdgeProblem, TryEdgeProblem, solve_edge_problem};
pub use dataflow::ifds::{
    CfgSupergraph, IfdsConfig, IfdsProblem, IfdsResults, Supergraph, solve_ifds,
};
pub use dataflow::instruction::InstrInfo;
pub use dataflow::node_fixpoint::{NodeProblem, TryNodeProblem, solve_node_problem};
pub use dataflow::program_point::ProgramPoint;
pub use dataflow::resume::{Resume, ResumeMode, resume_edge_problem, resume_node_problem};
pub use dataflow::slot_fixpoint::{
    OrderedFixpoint, SequentialExecutor, SlotAccess, WaveExecutor, WaveRequeue, ordered_fixpoint,
    wave_fixpoint,
};
pub use dataflow::solve::{
    Facts, IterationOrder, Seeds, Solve, SolveConfig, SolveError, SolveStats, TrySolveError,
    meet_options, try_meet_options,
};
pub use dataflow::stream::{stream_edge_problem, stream_node_problem};
pub use display::{DisplayInstr, IndentedWriter, render};
pub use edge::{Edge, EdgeId, EdgeKind, EdgeTag, KindedEdge, Probability};
pub use effects::{
    Access, AccessShape, CaptureBuilder, Captured, CapturedValue, DialectEffects, Effect, Expr,
    ExprId, Exprs, MemoryAddress, Opaque, RaiseOrigin, Raised, RtlEffect, SelectShape, StateTouch,
    StatementTree, TreeItem, TreeNode, TreeNodeId, TreeSpan, TreeSpanValue, Write, WriteKind,
};
pub use exception::{ExceptionDisposition, ExceptionFlow, ExceptionPhase};
pub use flow::{
    CallSite, EdgeRole, Flow, FlowControl, FlowEffect, Transfer, Transfers, UnresolvedRole,
    UnresolvedTransfer,
};
pub use graph::call_graph::{
    CallMetadata, FunctionIndex, FunctionNode, SummaryConfig, SummaryProblem, call_graph,
    call_graph_indexed, find_function, is_recursive_function, propagate_summaries,
    recursive_functions,
};
pub use graph::cdg::{
    ControlClosureScratch, NonterminationScratch, control_dependence_graph,
    nontermination_sensitive_controllers, nontermination_sensitive_controllers_in,
    weak_control_closure, weak_control_closure_in,
};
pub use graph::coldness::{ColdClass, ColdnessClasses};
pub use graph::cut::{EdgeCut, VertexCut, edge_disjoint_paths};
pub use graph::diff::{
    BlockHashes, BlockMatch, CascadeDiff, CfgDiff, MatchLevel, RegionDiff, RegionMatch,
};
pub use graph::dominator::{
    CertificateError, DominanceFrontiers, DominatorScratch, DominatorTree, DominatorUpdate,
    ExitPolicy, FrontierTree, LowHighOrder, PostDominatorTree, PostNode, SinkId,
};
pub use graph::dot::{
    DotEdgeAttributes, DotEdgeStyle, DotRankDir, DotStyle, plain_edge_attributes, to_dot, write_dot,
};
pub use graph::dyck::{BidirectedDyck, bidirected_dyck_classes};
pub use graph::edge_traverse::{
    breadth_first_edges, depth_first_edges, shortest_path_edges, shortest_path_edges_in,
};
pub use graph::edge_view::{EdgeRef, EdgeView, FilteredEdges};
pub use graph::eh::{EhBlockKind, EhModel};
pub use graph::epoch::EpochMarks;
pub use graph::fanout::{DenseFanout, Fanout, SortedMap};
pub use graph::feedback::{FeedbackConfig, feedback_arc_set, linear_arrangement};
pub use graph::fingerprint::{
    Mixer, canonical_hash, canonical_order, multiset_fingerprint, wl_colors,
};
pub use graph::frequency::{BlockFrequencies, FREQUENCY_SHIFT, FrequencyConfig};
pub use graph::horn::{HornClauses, HornSaturation};
pub use graph::keyed::KeyedGraph;
pub use graph::label::{Label, display_label, no_label};
pub use graph::loop_forest::{LoopForest, LoopId};
pub use graph::network_flow::{ArcId, FlowError, FlowNetwork, FlowSolution};
pub use graph::open::{
    FoldEnter, MarkScope, OpenBfsEvent, OpenDfsEvent, OpenFold, OpenFoldConfig, OpenFoldError,
    OpenFoldMemo, OpenPathsEvent, Revisit, follow, follow_path, open_breadth_first_events,
    open_breadth_first_paths, open_depth_first_events, open_fold, open_fold_in, open_search,
};
pub use graph::path_expr::{
    GenKill, GenKillAlgebra, PathAlgebra, PathExprArena, PathExprId, PathExprNode, PathExpressions,
    interpret, path_expressions,
};
pub use graph::profile_inference::InferenceCosts;
pub use graph::program_structure::{
    CycleEquivalence, ProgramStructureTree, SeseEdge, SeseRegionId,
};
pub use graph::reach_index::{ReachIndex, ReachIndexConfig, ReachScratch};
pub use graph::relax::min_label_relaxation;
pub use graph::scc::{
    IncrementalScc, Scc, SccChange, SccDecomposition, SccNumbering, StrongCutScratch, StrongCuts,
};
pub use graph::scope::{
    Scope, ScopeCursor, ScopeCursorPolicy, ScopeCursorQuery, ScopeDatum, ScopeDatumId, ScopeEdgeId,
    ScopeEdgeTag, ScopeEnvironmentCache, ScopeEnvironmentQuery, ScopeFirstResolution,
    ScopeFirstResolutionError, ScopeGraph, ScopeGraphChanges, ScopeGraphPathLabel, ScopeGraphQuery,
    ScopeId, ScopeIndexConfig, ScopeIndexUpdate, ScopePath, ScopeQuery, ScopeReference,
    ScopeReferenceId, ScopeResolution, ScopeResolutionCandidate, ScopeResolutionIndex, ScopeTag,
};
pub use graph::search::{
    DfsEvent, PathSearchConfig, PathSearchStats, SearchConfig, SearchOrder, SearchScratch, Visit,
    VisitedPolicy, depth_first_events, search, search_in,
};
pub use graph::stack::{
    StackEdge, StackEdgeId, StackEdgeTag, StackFileId, StackGraph, StackGraphError,
    StackLinearResolutionError, StackNode, StackNodeId, StackNodeKind, StackNodeTag,
    StackPartialPath, StackPartialPathDatabase, StackPartialPathId, StackPartialPathSet,
    StackPartialPathStats, StackPath, StackPathError, StackPathStep, StackResolution,
    StackResolutionIndex, StackScopedSymbol, StackSearchStats,
};
pub use graph::store::{
    AdjacentEdges, EdgeRecord, Graph, Id, IdTag, NodeId, NodeOrder, NodeTag, Renumbering,
};
pub use graph::structure::{BackEdge, find_back_edges, is_reducible};
pub use graph::text::{TextError, TextErrorKind, TextStyle, parse_text, to_text, write_text};
pub use graph::traverse::{
    CommonAncestor, ShortestPathScratch, breadth_first, common_ancestors, depth_first_postorder,
    depth_first_preorder, nearest_common_ancestor, reachable, reverse_postorder, shortest_path,
    shortest_path_in, topological_sort, topological_sort_or_cycle,
};
pub use graph::tree_decomposition::{EliminationHeuristic, TreeDecomposition};
pub use graph::verify::{VerifyError, VerifyReport, verify_edge_view, verify_view};
pub use graph::view::{
    DenseId, Direction, GraphView, Induced, NodeView, Reversed, Rooted, RootedView, VirtualRoot,
    scan_predecessors,
};
pub use graph::weak_topological_order::WeakTopologicalOrder;
pub use memory::{
    AccessMode, MemoryAccess, MemoryAccessKind, MemoryAtomicity, MemoryEvent, MemoryEventInfo,
    MemoryOperations,
};
pub use region::{
    Cleanup, CompletionReason, Continuation, ExclusiveExtent, ExtentIssue, ExtentPromotionStatus,
    Handler, HandlerBody, HandlerKind, HandlerMetadata, HandlerRef, Region, RegionError, RegionId,
    RegionIndex, promote_exclusive_extents, promote_handler_extents, recover_exclusive_extents,
};
pub use rewrite::Rewrite;
pub use transform::block_order::{ExtTspConfig, ext_tsp_order, ext_tsp_score, hot_cold_order};
pub use transform::cleanup::{
    merge_blocks, merge_blocks_mapped, remove_empty_blocks, remove_empty_blocks_mapped,
    remove_unreachable_mapped, simplify_mapped,
};
pub use transform::contract::{contract_edge, contract_edge_mapped};
pub use transform::critical::split_critical_edges_mapped;
pub use transform::layout::{RelaxError, RelaxMode, SpanItem, relax_layout, relax_spans};
pub use transform::linearize::{
    BlockOrder, Emitter, LinearInst, LinearizeError, falls_through, linearize,
};
pub use transform::loops::{CanonicalLoop, canonicalize_loops, insert_preheader};
pub use transform::pass::{
    Pass, PassError, PassExecution, PassFn, PassId, PassPipeline, PassReport, pass_fn,
};
pub use transform::reducible::{
    DispatchBuilder, DispatchReport, make_reducible, make_reducible_by_dispatch,
};
pub use union_find::DisjointSet;
