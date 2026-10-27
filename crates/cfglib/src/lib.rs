//! Generic graph and dataflow framework for code intelligence and program analysis.
//!
//! [`Graph`] is the one store: a compressed base plus an appendable delta,
//! holding arbitrary node and edge payloads for value-flow, symbol,
//! type-relation, import, call, and grammar graphs, with constant-time
//! addition, bit-cheap removal, identity-preserving endpoint redirection, and
//! an explicit [`compact`](Graph::compact) that reports its [`Renumbering`].
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
//! [`breadth_first_events`] and [`depth_first_events`] expose traversal-tree
//! structure for dense graphs; [`open_breadth_first_events`] and
//! [`open_depth_first_events`] provide the corresponding discovery streams
//! for lazily generated node spaces.
//! [`graph::scope::ScopeGraph`] stores language-defined scopes, labeled
//! reachability, relation-tagged declarations, and references.
//! [`graph::scope::ScopeGraphQuery`] supplies
//! the path automaton, matching, label/data specificity, or a full path order;
//! the same resolver handles stored references and ephemeral completion
//! requests and builds forward/reverse binding indexes.
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
//! [`ir`] holds the IR traits and the generic algorithms over them.
//! [`ir::rtl`] is the register-transfer contract: a write-only
//! [`ir::rtl::InstructionBuilder`] through which instruction semantics
//! describe one atomic state update, and read-only views over the sealed
//! records. [`ir::mlil`] is the read-only view contract of a flat static
//! single assignment function with block parameters; value analyses over it
//! belong to the consumer. [`ir::hlil`] is the structured contract: a
//! read-only [`ir::hlil::FunctionView`] of statement and expression trees and
//! a write-only [`ir::hlil::FunctionBuilder`] that the consumer implements,
//! with lift and lower between it and a flat variable-based function,
//! verification, pseudocode, declaration placement, and structure recovery
//! over them. Every IR storage, builder implementation, and editing
//! machinery belongs to the consumer; the algorithms here read through the
//! views, drive the builders, and report facts.
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
//! point. [`Cfg::to_text`] is the control-flow reading of it — block headers
//! with indented instructions, edges named by [`EdgeKind`], and exception
//! regions — and [`parse_cfg_text`] reads that back with a consumer-supplied
//! instruction parser. Every printer shares [`IndentedWriter`], so
//! indentation is decided once and a label is written straight into the sink.
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
//! [`CfgBuilder::build`]; [`resolve_jump_edges`] wires explicit gotos
//! afterwards via [`JumpTargets`].
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
//! FlowControl               (required by CfgBuilder and the tail-call detectors)
//!   └─ JumpTargets          (optional — explicit goto/label wiring, Target)
//!
//! InstrInfo<Variable = V>   (optional — native IR variables for dataflow)
//!   ├─ EffectInfo           (optional — side effects, Effect; purity + DCE)
//!   ├─ Predicated           (optional — guarded execution)
//!   ├─ CopySource           (optional — copy propagation)
//!   ├─ ConstantFolder       (optional — constant propagation, Const)
//!   ├─ ExprInstr            (optional — expression trees, Operator + Const)
//!   ├─ ValueNumberInfo      (optional — value numbering, Operator)
//!   └─ MemoryEventInfo      (optional — memory data flow + fences, Location + Fence)
//!
//! MemoryAlias<Location>     (optional may-alias oracle consumed by MemorySsa)
//!
//! DisplayInstr              (optional — rendering only: DOT, text)
//! CallInfo                  (optional — call graphs, Callee)
//! SwitchSource              (optional — switch table recovery, Target)
//! ```
//!
//! [`MemoryEventInfo`] is the sole instruction-side memory contract.
//! [`MemorySsa`] merges its reported locations through a caller-owned
//! [`MemoryAlias`] relation and provides location-class phis, reaching writes,
//! clobbers, users, transitive readers, and SSA-resolved address inputs.
//! [`MemoryValueFlow`] then combines matching ordinary and memory SSA forms in
//! one dependency graph whose typed edges distinguish addresses, stored and
//! loaded values, state reads/writes, and ordinary/memory phis.
//!
//! Additionally, [`Problem`] is the trait for pluggable instruction-level
//! dataflow analyses over any rooted view (run by [`solve_problem`]),
//! [`NodeProblem`] its node-level counterpart over any graph view (run by
//! [`solve_node_problem`]),
//! [`EdgeProblem`] its edge-sensitive counterpart (run by
//! [`solve_edge_problem`]), [`TryEdgeProblem`] the error-preserving edge
//! variant, and [`Emitter`] the trait for linearization output.
//!
//! # Contracts
//!
//! - Blocks may be empty, may lack explicit terminator instructions, and
//!   unreachable blocks are legal (dead code after a return/goto). SSA treats
//!   disconnected source components as independent dominator-forest roots, so
//!   their internal def-use flow remains intact without inheriting entry state.
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
//!   the old identity space). [`Rewrite::then_renumbered`] carries the first
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

pub use analysis::alias::AliasSets;
pub use analysis::control_tree::{
    BlockSpan, CatchHandler, ChildBodies, ControlBody, ControlDescent, ControlNode, ControlNodeId,
    ControlRun, ControlTree, ControlVisitor, GotoDiagnostic, GotoReason, LiftReport, LoopKind,
    Preorder, SwitchCase,
};
pub use analysis::dead_code::DeadCode;
pub use analysis::expr::{
    BlockExprTrees, ExprInstr, ExprNode, recover_block_expressions, recover_expressions,
};
pub use analysis::metrics::{
    CfgMetrics, GraphMetrics, block_nesting_depths, cfg_block_nesting_depths,
};
pub use analysis::pattern::{CfgPattern, detect_cfg_patterns, detect_patterns};
pub use analysis::purity::{Purity, block_purities, block_purity, cfg_purity};
pub use analysis::switch_table::{
    JumpTable, SwitchRecovery, SwitchSource, SwitchTargets, detect_switch_tables,
    recover_switch_tables,
};
pub use analysis::tail_call::{TailCall, detect_explicit_tail_calls, detect_tail_calls};
pub use analysis::value_numbering::{
    BlockValueNumbers, ValueNumber, ValueNumberInfo, ValueNumbering,
};
pub use block::{BasicBlock, BlockId, BlockTag};
pub use builder::address::{
    AddressBuildError, AddressCfgOptions, AddressEdgeInfo, AddressGraph, AddressHandler,
    AddressInstruction, AddressSpace, CallPolicy, HandlerChain, build_address_cfg,
    build_address_cfg_with_chains,
};
pub use builder::{BuildError, CfgBuilder, JumpResolution, resolve_jump_edges};
pub use cfg::{Cfg, CfgEdge, CfgRenumbering, SplitPointError, parse_cfg_text};
pub use dataflow::abstract_interpretation::{AbstractDomain, AbstractFacts, Lattice};
pub use dataflow::bits::DenseBits;
pub use dataflow::cfg_view::CfgView;
pub use dataflow::constant_propagation::{
    ConstFact, ConstPropProblem, ConstValue, ConstantFolder, ConstantPropagation,
};
pub use dataflow::copy_propagation::{
    AliasPairs, CopySource, PropagationStats, alias_propagation, alias_propagation_with_exits,
    copy_propagation, copy_propagation_with_exits,
};
pub use dataflow::def_use::DefUseChains;
pub use dataflow::edge_fixpoint::{
    EdgeFacts, EdgeProblem, Reachable, ReachableEdgeProblem, TryEdgeProblem, solve_edge_problem,
    solve_edge_problem_from, solve_edge_problem_from_with_config, solve_edge_problem_with_config,
    try_solve_edge_problem, try_solve_edge_problem_from, try_solve_edge_problem_from_with_config,
    try_solve_edge_problem_with_config,
};
pub use dataflow::fixpoint::{
    Direction, Facts, Problem, SolveConfig, SolveError, TryProblem, TrySolveError, meet_options,
    solve_problem, solve_problem_from, solve_problem_from_with_config, solve_problem_with_config,
    try_solve_problem, try_solve_problem_from, try_solve_problem_from_with_config,
    try_solve_problem_with_config,
};
pub use dataflow::instruction::{EffectInfo, InstrInfo, Predicated, VariableId};
pub use dataflow::liveness::{Liveness, LivenessProblem, SeededLivenessProblem};
pub use dataflow::memory::{
    ConservativeMemoryAlias, ExactMemoryAlias, MemoryAlias, MemoryClassId, MemoryDefinition,
    MemoryEventSite, MemoryLocationClass, MemoryPhi, MemorySsa, MemorySsaEvent, MemorySsaScratch,
    MemorySsaValue, MemoryUse, MemoryValueEdge, MemoryValueFlow, MemoryValueFlowError,
    MemoryValueFlowScratch, MemoryValueNode, MemoryValueRole, index_paths_may_overlap,
};
pub use dataflow::node_fixpoint::{
    NodeProblem, TryNodeProblem, solve_node_problem, solve_node_problem_from,
    solve_node_problem_from_with_config, solve_node_problem_with_config, try_solve_node_problem,
    try_solve_node_problem_from, try_solve_node_problem_from_with_config,
    try_solve_node_problem_with_config,
};
pub use dataflow::phi_web::{PhiWeb, PhiWebs};
pub use dataflow::program_point::{DefSite, ProgramPoint, UseSite};
pub use dataflow::reaching::{ReachingDef, ReachingDefs, ReachingDefsProblem};
pub use dataflow::sccp::SccpAnalysis;
pub use dataflow::ssa::{
    DominanceFrontiers, PhiPlacement, PhiPlacements, SsaBlock, SsaForm, SsaInstruction, SsaPhi,
    SsaScratch, SsaValue, SsaVersion,
};
pub use dataflow::ssa_destruction::{
    PhiCopy, PhiCopyGroup, PhiCopyOrigin, copies_by_predecessor, eliminate_phis,
};
pub use dataflow::unwind::departs_before_throws;
pub use display::{DisplayInstr, IndentedWriter};
pub use edge::{Edge, EdgeId, EdgeKind, EdgeTag, KindedEdge};
pub use exception::{
    ClrExceptionRegion, ClrHandler, ClrHandlerKind, ExceptionDisposition, ExceptionFlow,
    ExceptionPhase, SehExceptionRegion, SehHandler, SehHandlerKind, SehRegistration,
    SehRegistrationChain, VectoredExceptionModel, VectoredHandler, VectoredHandlerId,
    VectoredHandlerKind, VectoredHandlerOrder, VehModel, install_clr_region, install_seh_region,
};
pub use flow::{
    CallInfo, CallSite, EdgeRole, Flow, FlowControl, FlowEffect, JumpTargets, Transfer, Transfers,
    UnresolvedRole, UnresolvedTransfer,
};
pub use graph::call_graph::{
    CallMetadata, FunctionNode, call_graph, find_function, is_recursive_function,
    propagate_summaries,
};
pub use graph::cdg::control_dependence_graph;
pub use graph::coldness::{ColdClass, ColdnessClasses, ColdnessGraph};
pub use graph::diff::{BlockFingerprint, BlockMatch, CfgDiff};
pub use graph::dominator::{
    DominatorScratch, DominatorTree, FrontierScratch, PostDominatorScratch, PostDominatorTree,
    PostNode,
};
pub use graph::dot::{
    DotEdgeAttributes, DotEdgeStyle, DotRankDir, DotStyle, bind_edge_attributes,
    control_flow_edge_attributes, plain_edge_attributes, to_dot, write_dot,
};
pub use graph::edge_traverse::{
    EdgeStep, breadth_first_edges, breadth_first_edges_with, depth_first_edges,
    depth_first_edges_with, shortest_path_edges,
};
pub use graph::edge_view::{EdgeRef, EdgeView, FilteredEdges};
pub use graph::eh::{EhBlockKind, EhEdge, EhEdgeKind, EhModel};
pub use graph::epoch::{EpochMarks, EpochSet};
pub use graph::fanout::{DenseFanout, Fanout, SortedMap};
pub use graph::horn::HornClauses;
pub use graph::interval::{Interval, IntervalAnalysis};
pub use graph::keyed::KeyedGraph;
pub use graph::label::{Label, bind_label, display_label, no_label};
pub use graph::loop_forest::{LoopForest, LoopId};
pub use graph::loop_nest::{LoopNestNode, LoopNestingTree};
pub use graph::open::{
    FoldEnter, MarkScope, OpenBfsConfig, OpenBfsEvent, OpenDfsConfig, OpenDfsEvent, OpenFold,
    OpenFoldConfig, OpenPathsConfig, OpenPathsEvent, OpenSearchConfig, follow, follow_path,
    open_breadth_first_events, open_breadth_first_paths, open_depth_first_events,
    open_fold_post_order, open_search,
};
pub use graph::pdg::{DependenceKind, DependenceNode, program_dependence_graph};
pub use graph::reducible::make_reducible;
pub use graph::relax::min_label_relaxation;
pub use graph::reverse::reverse_cfg;
pub use graph::scc::{
    Scc, SccDecomposition, condensation, condensation_of, kosaraju_scc, tarjan_scc,
};
pub use graph::scope::{
    Scope, ScopeDatum, ScopeDatumId, ScopeEdgeId, ScopeEdgeTag, ScopeGraph, ScopeGraphPathLabel,
    ScopeGraphQuery, ScopeId, ScopeLinearResolutionError, ScopePath, ScopeQuery, ScopeReference,
    ScopeReferenceId, ScopeResolution, ScopeResolutionCandidate, ScopeResolutionConfig,
    ScopeResolutionIndex, ScopeResolutionStats, ScopeTag,
};
pub use graph::search::{
    BfsEvent, DfsEvent, SearchConfig, SearchOrder, SearchScratch, Visit, VisitedPolicy,
    breadth_first_events, depth_first_events, search, search_with_marks, search_with_scratch,
};
pub use graph::stack::{
    StackEdge, StackEdgeId, StackEdgeTag, StackFileId, StackFileTag, StackGraph, StackGraphError,
    StackLinearResolutionError, StackNode, StackNodeId, StackNodeKind, StackNodeTag,
    StackPartialPath, StackPartialPathConfig, StackPartialPathDatabase, StackPartialPathId,
    StackPartialPathSet, StackPartialPathStats, StackPath, StackPathError, StackPathStep,
    StackResolution, StackResolutionIndex, StackReverseIndex, StackScopedSymbol, StackSearchConfig,
    StackSearchStats,
};
pub use graph::store::{AdjacentEdges, EdgeRecord, Graph, Id, IdTag, NodeId, NodeTag, Renumbering};
pub use graph::structure::{
    BackEdge, CanonicalLoop, NaturalLoop, canonicalize_loops, detect_loops, detect_loops_tagged,
    find_back_edges, find_back_edges_tagged, insert_preheader, is_reducible, loop_exit_blocks,
};
pub use graph::text::{TextError, TextErrorKind, TextStyle, parse_text, to_text, write_text};
pub use graph::traverse::{
    CommonAncestor, TraversalDirection, breadth_first, common_ancestors, depth_first_postorder,
    depth_first_preorder, nearest_common_ancestor, reachable, reverse_postorder, shortest_path,
    topological_sort,
};
pub use graph::verify::{
    SemanticValidator, SemanticVerifyReport, VerifyError, VerifyReport, verify, verify_edge_view,
    verify_view, verify_with,
};
pub use graph::view::{
    DenseId, GraphView, NodeView, Reversed, Rooted, RootedView, scan_predecessors,
};
pub use memory::{
    MemoryAccess, MemoryAccessKind, MemoryAtomicity, MemoryEvent, MemoryEventInfo,
    MemoryOperations, MemoryTrace, MemoryTraceEntry,
};
pub use region::{
    Cleanup, CompletionReason, Continuation, ExclusiveExtent, ExtentIssue, ExtentPromotionDecision,
    ExtentPromotionStatus, ExtentStatus, Handler, HandlerBody, HandlerFilters, HandlerKind,
    HandlerMetadata, HandlerRef, HandlerTypes, Region, RegionId, RegionIndex,
    promote_exclusive_extents, promote_handler_extents, recover_exclusive_extents,
    recover_exclusive_extents_with,
};
pub use rewrite::Rewrite;
pub use transform::cleanup::{
    merge_blocks, merge_blocks_mapped, remove_empty_blocks, remove_empty_blocks_mapped,
    remove_unreachable, remove_unreachable_mapped, simplify, simplify_mapped,
};
pub use transform::coloring::{ColorAssignment, color_graph, interference_graph};
pub use transform::contract::{contract_edge, contract_edge_mapped};
pub use transform::critical::{
    split_critical_edges, split_critical_edges_mapped, split_critical_edges_with,
};
pub use transform::dce::{
    dead_code_elimination, dead_code_elimination_with_exits, remove_dead_code,
    remove_dead_code_mapped,
};
pub use transform::layout::{RelaxError, relax_layout};
pub use transform::linearize::{BlockOrder, Emitter, LinearInst, LinearizeError, linearize};
pub use transform::loops::{LoopRotation, find_loop_invariants, rotate_loop};
pub use transform::pass::{
    Pass, PassChange, PassExecution, PassFailure, PassFn, PassId, PassPipeline, PassReport, pass_fn,
};
pub use transform::pre::{PreAnalysis, eliminate_pre};
pub use union_find::DisjointSet;
