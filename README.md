# cfglib

Generic, `no_std` graph and dataflow framework for code intelligence, program analysis, decompilation, and compiler infrastructure.

`cfglib` has purpose-built storage for arbitrary directed relations, language-parametric scope graphs, file-incremental stack graphs, and `Cfg<I, E = ()>` graphs that genuinely need basic blocks, typed control-flow edges, caller-owned edge metadata, and exception regions. Its generic RTL and MLIL layers combine CFGs with checked construction, exact edge payloads, signatures, exception regions, and many-to-many source provenance; RTL/MLIL bridges recover typed semantic variables from native storage or lower them back while allowing the two levels to use distinct dialect and native-location types. Small read-only node and edge view contracts let consumer-owned stores and zero-copy filtered views reuse the algorithms. Every instruction-adjacent axis is consumer-typed rather than imposed by the library: dataflow variables, constants, expression operators, side-effect vocabularies, branch targets, call targets, and edge provenance all come from the adapter — so x86 registers and flags, shader register components, bytecode locals, compiler IR values, and source-language symbols do not need to be flattened into a library-owned numbering scheme, and string literals or symbol ids flow through the analyses as naturally as machine words and addresses. On top of that it ships a compiler-middle-end toolkit: dominator trees, renamed SSA construction, location-aware memory SSA, dataflow analyses, value numbering, explicit alias sets, loop transforms, dead-code elimination, partial redundancy elimination, graph coloring, and structured AST recovery.

Named transformations compose through `PassPipeline<T, E>`. A pipeline keeps
the caller's declared order, supports closure-backed or stateful trait passes,
reports which executions changed the target, and attributes the first failure
while retaining the completed-prefix report. Pass selection and canonical
schedules remain consumer policy, so one framework can drive CFG, RTL, MLIL,
HLIL, or consumer-owned compilation contexts without coupling their dialects.

Everything is `no_std + alloc` and the core graph structure uses `SmallVec` adjacency lists with tombstone-based edge removal for cache-friendly, arena-stable IDs.

## Quick start

### Direct construction (the primary front door)

No trait is required to build, verify, analyze, or render a CFG. Source
frontends lower their syntax trees straight into blocks:

```rust
use cfglib::{Cfg, DominatorTree, EdgeKind, ReachingDefs, verify};

let mut cfg = Cfg::<Stmt>::new();
let then_block = cfg.new_block();
let merge = cfg.new_block();
cfg.block_mut(cfg.entry()).push(stmt_if);
cfg.block_mut(then_block).push(stmt_assign);
cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);

assert!(verify(&cfg).is_ok());
let dominators = DominatorTree::compute(&cfg);
let reaching = ReachingDefs::compute(&cfg); // needs InstrInfo on Stmt
```

### Builder construction (structured instruction streams)

Frontends with a flat, structured stream (shader bytecode, structured ISAs)
implement `FlowControl` and use `CfgBuilder`; explicit gotos are wired
afterwards through the opt-in `JumpTargets` trait:

```rust
use cfglib::{CfgBuilder, FlowControl, FlowEffect, resolve_jump_edges};

impl FlowControl for Inst {
    fn flow_effect(&self) -> FlowEffect {
        match self.opcode {
            Op::If    => FlowEffect::ConditionalOpen,
            Op::Else  => FlowEffect::ConditionalAlternate,
            Op::EndIf => FlowEffect::ConditionalClose,
            Op::Loop  => FlowEffect::LoopOpen,
            Op::End   => FlowEffect::LoopClose,
            Op::Ret   => FlowEffect::Return,
            _         => FlowEffect::Fallthrough,
        }
    }
}

let mut cfg = CfgBuilder::build(instructions).unwrap();
let resolution = resolve_jump_edges(&mut cfg); // wires goto/label edges
```

## Feature overview

### Generic graph core

`DirectedGraph<N, E>` is the single owned storage type for consumer-defined node and edge payloads, stable IDs, and forward/reverse adjacency. `DirectedGraphView` lets existing graph stores use node algorithms without first migrating their storage; `EdgeGraphView` additionally exposes stable edge identity, view-oriented endpoints, payloads, and ordered adjacency. `FilteredEdges` borrows either representation through an edge predicate without cloning or renumbering anything—for example, the same CFG can be viewed as normal-only flow or full normal-plus-exception flow. `RootedGraphView` adds a distinguished entry node for algorithms that need one (dominance, reachability metrics, loop and interval analysis), and the `Rooted` adapter roots any plain view at a chosen node. This layer is suitable for symbol/reference graphs, value-flow graphs, call graphs, type relations, import graphs, grammar dependencies, and analysis-derived relations; it has no instruction or binary-analysis concepts.

```rust
use cfglib::{DirectedGraph, Rooted, DominatorTree, TraversalDirection, shortest_path};

let mut graph = DirectedGraph::new();
let source = graph.add_node("definition");
let target = graph.add_node("call result");
let edge = graph.add_edge(source, target, ("return", "src/lib.rs", 42));

assert_eq!(graph[edge].payload().0, "return");
assert_eq!(
    shortest_path(&graph, source, target, TraversalDirection::Outgoing),
    Some(vec![source, target]),
);
let dominators = DominatorTree::compute(&Rooted::new(&graph, source));
```

### Scope graphs

`graph::scope::ScopeGraph<S, L, R, D, Q>` separates language facts from
language policy. It stores consumer-defined scope metadata, labeled
scope-to-scope edges, relation-tagged declarations, and references. A `ScopeGraphQuery`
implementation supplies the path-language state machine, declaration matching,
label and data orders, optional indexed candidate visitation and decisive-match
pruning, and—when needed—a full path/data shadowing order. This keeps lexical parents, imports,
inheritance, namespaces, ordered declarations, accessibility, redirects, and
opacity in the frontend's vocabulary.

`ScopeResolution::compute` resolves a stored reference.
`ScopeResolution::compute_from` runs the same algorithm for an ephemeral query,
which supports completion and speculative refactoring without modifying the
graph. `ScopeResolutionIndex` resolves all stored references and provides the
reverse definition-to-reference relation used by rename and find-references.
Paths are cycle-free, deterministic, and optionally bounded; truncated results
report which bound was reached. Scope storage supports `serde` when the feature
is enabled.

### Stack graphs

`graph::stack::StackGraph<F, S, N, E>` implements the standard root,
exported/internal scope, symbol push/pop, scoped-symbol push/pop, drop-scopes,
and jump-to-scope nodes. Symbols, file records, node metadata, and edge metadata
are consumer types.
Concrete paths maintain symbol and scope stacks transactionally, scoped symbols
can pause one lookup while another resolves, and edge precedence removes
shadowed complete paths while retaining genuine ambiguity.

Every ordinary node belongs to one file partition. Direct cross-file edges are
rejected; independently constructed files compose through the root or exported
scope stitching endpoints. `StackPartialPathSet` extracts reusable structural
routes per file, `StackPartialPathDatabase` replaces one file at a time while
leaving other identities stable, and `StackGraph::clear_file` retires one
changed partition with node/edge tombstones before it is rebuilt. Then
`StackResolution::compute_from_partials` replays compatible summaries with the
same semantics as direct search. Graphs and partial-path databases are
serializable with `serde`, so unchanged file summaries can be persisted and
reused. Both direct and stitched indexes expose reverse
definition-to-reference bindings.

### Control-flow graph (`Cfg<I>`)

| Feature | Description |
|---|---|
| Generic `Cfg<I, E = ()>` | Parameterised over any instruction and caller-owned edge payload; unit payload preserves the original API |
| `no_std` + `alloc` | Runs in embedded, kernel, and WASM environments |
| `CfgBuilder` | Builds a CFG from a flat structured instruction stream (`if/else/endif`, `loop/endloop`, `switch/case/endswitch`, `break`, `continue`) |
| Goto wiring | `resolve_jump_edges` + `JumpTargets` (consumer-typed targets: labels, addresses, syntax nodes) |
| SmallVec adjacency | Stack-allocated successor (2) / predecessor (4) lists; heap only for high fan-out |
| Tombstone edges | `remove_edge()` replaces the slot with `None`; existing `EdgeId`s remain stable |
| Edge metadata | Stable `EdgeId`, endpoints, `EdgeKind`, optional weight, and a caller-owned payload; `ExceptionFlow<M>` standardises search/unwind phase and execute/search/continue disposition while retaining platform metadata |
| Regions | Try/catch/finally regions with explicit `HandlerBody::Known`/`Unknown` completeness and `HandlerKind` (Catch, CatchAll, Finally, Fault, Filter); `install_clr_region` and `install_seh_region` import normalized platform clauses |
| Cleanup continuations | `add_continuation` / `set_cleanup_resume` → `Cleanup` (`Continuation` + `CompletionReason`): which route out of a single shared `finally` block a resume edge belongs to |
| Handler metadata | `HandlerMetadata<M>` side table (`HandlerFilters<F>` / `HandlerTypes<T>`) — consumer-typed filter predicates and CLR caught-type tokens keyed by `HandlerRef`, with no type parameter on `Cfg` |
| Native handler state | `SehRegistrationChain<F, H>` models per-thread x86 frame registrations; `VehModel<H>` separately models ordered process-wide vectored exception/continue handlers |
| Subgraph extraction | `subgraph()` or `subgraph_mapped()` with dense O(1) block-id remapping and payload cloning |
| Block splitting | `split_block()`, mapped payload-aware variants, and validated multi-point splitting with automatic stable edge transfer |
| `serde` feature | Optional serialization support |

| Leader-based construction | `build_address_cfg`, `AddressCfgOptions` / `CallPolicy`, `AddressInstruction` / `AddressSpace` traits, `Flow`, `AddressHandler` | Machine/bytecode streams: caller-selected function entry; calls kept as edges or flattened into reported call sites; leaders at targets, after terminators, at exceptional instructions, and at exception boundaries; known successors retained beside explicit unresolved transfers; typed normal, instruction-exceptional, and unwind edges through a consumer payload hook, every kind chosen by `EdgeRole::kind`; nested regions registered enclosing-first with parents wired |
| Structured-marker walk driver | `StructuredWalk` + `StructuredSink` | The if/else/loop/break/continue frame bookkeeping for lifts that emit into checked builders (RTL, MLIL) or any block store; typed `StructuredWalkIssue` misuse reporting |

### Generic register-transfer IR (`ir::rtl`)

`ir::rtl::Function<D>` stores typed expression reads and parallel transfers over
consumer-defined native locations. Its checked builder retains stable statement
identities, ordered parameters and returns, exception regions, exact caller edge
payloads, and deterministic many-to-many source provenance.

`MlilBridge` associates an RTL dialect with a semantic MLIL dialect without
requiring the dialect markers or native-location types to match. `lift()` uses
lane SSA and live phi webs to recover typed semantic variables, then performs
fallible edge translation only after every statement-to-instruction mapping is
known. `lower()` applies consumer placement and instruction selection in the
opposite direction. Both return rewrite maps, translate signatures and regions,
preserve instruction expansion/fusion provenance, and keep exceptional throw
sites exact. The consumer explicitly maps native RTL storage into optional MLIL
provenance, so target allocation and synthetic temporaries are not mistaken for
source locations.

| Template constant folding | `VarExpr::fold_constant`, `Shape::holds_words` | Bitwise evaluation of value templates: positional read resolution and shape checking are generic; only the operator semantics hook stays dialect-owned |
| Whole-place storage planning | `plan_whole_places` | The standard variable-shaped plan: preserved native slots keep their storage, everything else mints dense generated slots, every place spans lane zero |

### Generic medium-level IR (`ir::mlil`)

`ir::mlil::Function<D>` is a verified semantic function over the same payload-aware
CFG used by the rest of cfglib. The `Dialect` parameter supplies the operation,
value-type, effect, edge, source-coordinate, variable-role, and native-variable
types. `AnalysisDialect` adds constant folding, pure-expression, copy, and call
hooks; `VerifyDialect` appends semantic invariants after cfglib checks the
generic graph, identities, def/use tables, edge classification, and provenance.

`ir::mlil::FunctionBuilder<D>` assigns dense stable instruction and variable IDs,
retains source expansion and fusion through `ProvenanceMap<D>`, records the
ordered parameter/return `Signature<D>` and validated exception regions
(`add_region`, `HandlerBody::Unknown` allowed), and exposes a function only
after verification. The resulting function directly provides dominance, SSA,
def-use, liveness, constant and expression recovery, copy-propagated and
dead-code-eliminated views, identity-ordered instruction iteration, and
structured control flow with a fidelity report. SSA stays a derived view over
the canonical function; SSA-shaped transforms are expressed as
view-then-rebuild passes — `split_variables()` computes the function's own
SSA, partitions values into phi webs, and rebuilds the function with one
variable per lifetime (blocks, edges, instructions, and regions keep their
identities; both variable mappings are returned), so storage-derived slot
reuse separates into clean per-lifetime locals before HLIL lifting. The
partition uses liveness-pruned phi webs, so a dead phi at an unpruned join
never unites unrelated lifetimes. `with_promoted_handler_extents()` returns
a derived function whose `HandlerBody::Unknown` extents are promoted to
their dominated blocks, so extent-dependent consumers (structured `try`
recovery, HLIL lifting) run while the canonical function keeps stating the
extents are unknown. `with_duplicated_structuring_tails()` copies small
shared tails (short-circuit conditions, shared side-exits) until control
flow tree-structures — a pure unfolding for presentation views, driven by
the structuring report and refused for loop headers, region members, and
exceptional flow. `extend_equivalent_coverage()` grows each exception
region's protected set with provably equivalent coverage — a block that
cannot throw whose sequential predecessors are all protected joins, to a
fixpoint — restoring the construct-shaped coverage compilers trim at the
last throwing instruction. `with_derived_cfg(transform)` is the general
form of the same door: it clones the function and hands the consumer the
clone's CFG for dialect-aware presentation surgery (detaching a runtime's
self-covering cleanup ranges) — the canonical function is never touched.

Memory stays outside variables by contract: anything aliasable lives behind
dialect load/store operations ordered by their declared effects, while
variables are unaliasable dataflow-visible storage. `promote_memory()`
(`PromoteDialect`) moves values across that boundary soundly: the consumer
classifies which instructions access which statically-fixed locations and
which disqualify them (address taken, opaque reach), and the library
rewrites every access of an unaliased location into copies through one
fresh variable — same identity-preserving rebuild, composing with
`split_variables` and HLIL lifting (promote, split, lift: frame slots
become typed locals).
No language, runtime, calling convention, opcode, or source-location type is
built into the representation; those remain in the dialect crate.

### Generic high-level IR (`ir::hlil`)

`ir::hlil::Function<D>` is the structured, expression-oriented level above
MLIL: statements form trees (assignments, `if`, `while`/`do-while`/`loop`/`for`,
`switch` over constant case values, `try` with typed handler arms, labeled
statements and `goto` residue, and a dialect-defined `Region` statement for
shapes like `synchronized`/`using`), while values nest as typed expression
trees over the dialect's open operation vocabulary. Every expression is one
typed occurrence with exactly one parent, and verification enforces tree
shape, label resolution, and transfer contexts.

The level-independent vocabulary (`Vocabulary`: value types, effects, source
coordinates, variable roles) is shared with `ir::mlil::Dialect`, so one
consumer dialect type serves both levels. Three doors move functions through
HLIL:

| Door | Description |
|---|---|
| `FunctionBuilder<D>` | Checked bottom-up construction for source-language lowering (`add_expression` / `add_statement` / `set_body`) |
| `lift_function(&mlil::Function<D>)` | Binary/bytecode lifting: structures control flow via `ir::ast`, recognizes `while`/`do-while` conditions, recovers switch case values from dispatch-edge payloads, structures declared exception regions, empties jump trampolines in its working view (blocks whose every instruction is a dialect-declared pure transfer — `Lifted::ControlFlow`, no definitions, no throw — so `break`/`continue` resolutions forward through them), and inlines single-use definitions into expression trees while provably preserving effect and exception order — with `LiftDialect::evaluation_commutes` letting the dialect widen that order (read-read pairs fold into one expression; the default refuses every pair). Straight-line block runs coalesce into one translation list, so frontends that emit one block per native instruction still inline across the whole linear region, and a single-pair parallel move (a type-refinement pair, a lone phi-copy commit) inlines as a plain copy. `lift_function_with_metadata` can omit instruction correspondence and provenance for semantics-only consumers |
| `lower_function(&hlil::Function<D>)` | The downward mirror: statements flatten to blocks and edges with lazy join materialization, expression trees linearize into typed temporaries, loops/switches/labels wire their transfers, and declared `try` regions register with known handler extents and unwind edges — producing a verified `mlil::Function` for flat analyses or consumer code generation |

Both level bridges return per-instruction maps between MLIL identities and
the HLIL entities carrying them, and compose source provenance across the
translation, so source maps survive in either direction.

Consumers walk the trees without matching every variant:
`StatementKind::child_bodies` yields each nested statement sequence in
execution order, `StatementKind::for_each_expression` visits the statement's
own operands, `Function::visit_statements` walks statement trees in
preorder, and `Function::visit_expression_tree` walks one expression tree
in postorder — so a new statement or node shape extends downstream walkers
automatically. `Function::expressions_equal` compares expression trees
structurally, and `Function::compound_assignment` recognizes
`target = op(target, operand)` for renderers that spell `target op= operand`
— by structure, never by comparing rendered text.

Above the lift, `recover_structure` (`RecoverDialect`) rebuilds a function
with source-level shapes restored: `init; while (c) { …; update }` becomes
a `for` statement, assigning and returning diamonds become the dialect's
`select` expression (evaluating one arm, like `?:`), and paired enter/exit
protocols with their exceptional cleanup handlers become `Region`
statements (`synchronized`, `lock`) — each recovery opt-in per dialect
hook, each a pure re-expression, and each remappable through the returned
identity maps (`LiftedFunction::with_recovered_structure` composes it with
the lift's instruction table). `RecoverDialect::single_expression_operation`
keeps operations that expand into statement sequences out of `for` clauses
and selection arms. Exit-on-true loop conditions negate exactly
through `LiftDialect::negate_operation` — a comparison with its relation
inverted — before falling back to a wrapping `logical_not`.

### Graph algorithms

| Algorithm | Function / Type | Description |
|---|---|---|
| DFS / BFS | `depth_first_preorder`, `depth_first_postorder`, `breadth_first`, and matching `Cfg` methods | Direction-selectable traversals over `DirectedGraphView`; abbreviated `Cfg::dfs_*` / `Cfg::bfs` methods remain compatibility aliases |
| Edge-aware traversal | `breadth_first_view_edges[_with]`, `depth_first_view_edges[_with]`, `shortest_path_view_edges`, and matching owned-graph wrappers | Every distinct edge once with identity + endpoints over any edge view; parallel-edge provenance; `walk_*` names remain breadth-first compatibility aliases |
| Configurable search | `search` + `SearchConfig` (order, visited policy, direction, depth bound) | First-match, pruning (`Visit::Skip`), early exit (`ControlFlow::Break`), and backtracking as configuration; `VisitedPolicy::Path` un-marks on unwind so every route to a node is reported |
| Reusable search marks | `search_with_marks` + `EpochMarks` | The same search with its visited marks in a caller-owned epoch-stamped buffer: a per-root pass allocates marks once instead of an O(node count) buffer per root, and each search still starts from a clean set (epoch bump, O(1)) |
| Reusable search scratch | `search_with_scratch` + `SearchScratch` | The same search with the marks *and* the call's own buffers — seeds, frontier, adjacency — caller-owned, for a pass whose searches are small enough that the call is the cost; marks and buffers both reset on entry |
| Disjoint sets | `DisjointSet` (`union` by rank, `union_toward_min` for deterministic minimum representatives, growable via `push`) | The union-find backing phi webs, may-alias classes, and lane webs, usable directly by consumers |
| Open post-order fold | `open_fold_post_order` + `OpenFold` trait (`FoldEnter::Leaf`/`Fold`, `MarkScope::Shared`/`Isolated`) | Child-answers-to-parent folding over a lazily discovered space: caller-keyed cycle guard (nodes need no `Ord`), per-node mark scoping (exact per-path marking or persistent pruning), early exit, depth bound — C++-style member lookup and type-algebra evaluators without hand-written frame stacks |
| Forward-only view support | `scan_predecessors` | The honest O(nodes × edges) `predecessors` for views that store only successors, replacing per-consumer scan stubs |
| Traversal events | `breadth_first_events` → `BfsEvent`; `depth_first_events` → `DfsEvent` | Breadth-first discovery with tree/non-tree edges, or depth-first discover/tree/back/forward-or-cross/finish events in pinned order |
| Open-graph search | `open_search` + `OpenSearchConfig` | The same disciplines over a lazily discovered node space: successors come from a closure, no dense ids (import/re-export chases, ordered emission walks) |
| Open-graph events | `open_breadth_first_events` → `OpenBfsEvent`; `open_depth_first_events` → `OpenDfsEvent` | Breadth-first discovery/refusal events or depth-first discover/finish/refusal events over a lazily discovered node space; path policy can revisit a shared node once per route |
| Open-graph route enumeration | `open_breadth_first_paths` → `OpenPathsEvent` | Simple routes shortest-first over a lazily discovered space — no global marks, per-route cycle guard; exponential worst case, so bound with depth, `Skip`, or `Break` |
| Alias chase | `follow`, `follow_path` | Out-degree ≤ 1 chase with a hop bound and a full-path cycle guard; the chain, or just its end |
| Shortest path | `shortest_path` (nodes), `shortest_path_edges` (edge witness) | Forward or reverse unweighted witness path |
| Minimum-label relaxation | `min_label_relaxation` | Edge-defined label transfer to a minimum fixpoint; nodes re-expand when their label improves |
| Multi-source reachability | `reachable` | Dense `Vec<bool>` from a seed set, forward or reverse; order-insensitive |
| Nearest common ancestor | `nearest_common_ancestor` | Bidirectional BFS meet of two nodes; smallest combined distance, ties by smallest node id |
| All common ancestors | `common_ancestors` → `Vec<CommonAncestor>` | Every shared node with both hop counts, depth-bounded, in `b`'s BFS discovery order — the consumer-rankable form (MRO linearization, overload preference) |
| Horn-clause derivability | `HornClauses` | AND-OR closure (`head <- b1 & b2`): nullability, all-arguments-constant, all-callers-dead |
| Topological sort | `topological_sort` | Stable ordering or cycle detection |
| Dominator tree | `DominatorTree::compute` (rooted views), `compute_from` (explicit root) | Cooper-Harvey-Kennedy over any graph view |
| Post-dominator tree | `DominatorTree::compute_post` (CFG), `compute_post_from` (any view + explicit exits) | Virtual-exit handling built in |
| Dominance frontiers | `DominanceFrontiers::compute` | For SSA φ-placement |
| Dominator recompute diff | `DominatorTree::compute_with_diff` | Recompute after a graph edit, reporting the nodes whose idom changed |
| Strongly connected components | `tarjan_scc` → `SccDecomposition<N>`, `condensation` → component DAG | Generic iterative Tarjan algorithm, reverse-topological order (leaves first) |
| SCC in topological order | `kosaraju_scc` → `SccDecomposition<N>` | The same partition numbered sources first (`index(u) < index(v)` across every edge); the classic deterministic two-pass algorithm, for budgeted forward closures over the condensation |
| Condensation of a given decomposition | `condensation_of(graph, &SccDecomposition)` → `DirectedGraph<(), ()>` | The component DAG whose node index **is** the given decomposition's component index, either algorithm's; deduplicated edges, and in-degrees plus dependents straight off the graph (the one-pass fixpoint shape) |
| Back-edge detection | `find_back_edges` (dominance, any view), `find_back_edges_tagged` (CFG, honors `Back` tags) | |
| Natural loop detection | `detect_loops` / `detect_loops_tagged` → `Vec<NaturalLoop<N>>` | Header, body, latches, nesting depth |
| Loop nesting tree | `LoopNestingTree::compute` | Parent/child loop hierarchy |
| Control dependence graph | `control_dependence_graph` → `DirectedGraph<N, ()>` | From post-dominator tree, over any view |
| Program dependence graph | `program_dependence_graph` → `DirectedGraph<DependenceNode, DependenceKind>` | Control + def-use edges; reverse traversal performs backward slicing |
| Interval analysis | `IntervalAnalysis::compute` | T1-T2 reduction over rooted views; reducibility test |
| Reducibility transform | `make_reducible` | Node splitting for irreducible CFGs |
| Reverse CFG | `reverse_cfg` | Flip all edges, swap entry/exits |
| Call graph | `call_graph` + `CallInfo`, `propagate_summaries` (callee-first SCC fixpoint) | Consumer-typed callees; interprocedural summary scaffold |
| CFG diff | `CfgDiff::compute` | Structural comparison (bindiff-style fingerprinting), no trait bounds |
| Exception handling model | `EhModel::compute` | Payload-generic CFG input; stable source `EdgeId`, exact handler/unwind/leave/resume/continue kinds, landing pads, cleanup/resume blocks, handler identities, protected-by mapping, and cleanup continuations |
| Integrity verification | `verify`, `verify_view`, `verify_edge_view`; `verify_with` + `SemanticValidator` | Structural node/edge-view checks plus deterministic typed consumer hooks for cardinality, ordering, and provenance rules |
| DOT export | `to_dot` (`DisplayInstr`), `to_dot_with` (bound-free), `write_view_dot` / `to_view_dot` (any view) | Graphviz output with escaped labels |

### Dataflow framework

| Analysis | Function / Type | Description |
|---|---|---|
| Generic fixpoint solver | `solve_problem[_from][_with_config]`, `Problem` / `TryProblem` traits | Forward or backward, any lattice type; every solver carries the same seeded/bounded/fallible matrix over shared `SolveConfig` and `SolveError`; `meet_options` lifts any lattice to `Option<F>` with `None` as unreachable bottom |
| Node-level fixpoint | `solve_node_problem[_from][_with_config]`, `NodeProblem` / `TryNodeProblem` traits | Per-node facts over any graph view (taint, reachability-with-facts) |
| Edge-sensitive fixpoint | `solve_edge_problem`, `solve_edge_problem_from`, `EdgeProblem` trait | Full or seeded per-edge transfer over any edge view; stable id/data plus physical node pre/post states and deterministic bounded-solve errors |
| Fallible edge-sensitive fixpoint | `try_solve_edge_problem`, `try_solve_edge_problem_from`, `TryEdgeProblem` trait | Preserves consumer boundary, merge, node-transfer, and edge-transfer errors separately from solver limits |
| Reachability-lifted edge analysis | `Reachable` + `ReachableEdgeProblem` trait | Verification-style analyses over `Option` facts: `None` is unreached bottom, transfers short-circuit, entry facts start the flow, and each edge chooses pre- or post-state (exceptional edges observe the state the node received) |
| Dense bit-set facts | `DenseBits` | Set-of-indices lattice element packed 64 per word; `union_with` is the join and reports change for convergence checks |
| Reaching definitions | `ReachingDefs::compute` | Which writes reach each point |
| Liveness | `Liveness::compute` | Live-in / live-out at each block; `live_before_instructions` / `live_after_instructions` replay the block transfer for instruction-granular sets (dead-store detection) |
| Def-use / use-def chains | `DefUseChains::compute` | Bidirectional def↔use links; dead-def detection |
| SSA construction | `SsaForm::compute` | IDF phi placement plus full dominator-forest renaming, including disconnected handler/dead-code components |
| Phi placement | `PhiPlacements::compute` | Structural IDF phase for consumers that only need placement |
| SSA deconstruction | `eliminate_phis`, `copies_by_predecessor` | φ-to-copy lowering |
| Phi webs | `PhiWebs::compute` | Congruence classes for register coalescing |
| Constant propagation | `constant_propagation`, `ConstantFolder` (associated `Const`) | Top/Const/Bottom lattice over a consumer constant domain — machine words, strings, bools, float bits |
| Sparse conditional constant propagation | `SccpAnalysis::compute` | SSA-based, marks unreachable edges |
| Copy and value-alias propagation | `copy_propagation`, `alias_propagation`, `CopySource` trait | Guarded chain resolution and dead transfer removal; pairwise aliases may refine types or metadata without changing runtime values |
| Memory-event trace | `MemoryTrace::compute`, `MemoryEventInfo` trait | Ordered, location-typed reads, writes, read/modify/write accesses, address-variable dependencies, and fences; instruction summaries distinguish separate read+write from compound modification |
| Memory SSA | `dataflow::memory::MemorySSA::compute`, `MemoryAlias` trait | Event-driven SSA per may-alias location class: loop/branch φ-nodes, reaching writes and clobbers, bidirectional def-use chains, transitive readers, and ordinary-SSA address inputs |
| Memory value flow | `dataflow::memory::MemoryValueFlow::compute` | One graph over ordinary SSA values, versioned memory states, and exact events; typed address/store/read/write/load and ordinary/memory-phi edges retain the complete transfer path |
| Abstract interpretation | `abstract_interpret`, `AbstractDomain` trait | Generic abstract domain framework |

### Higher-level analyses

| Analysis | Function / Type | Description |
|---|---|---|
| Expression tree recovery | `recover_expressions`, `ExprInstr` (associated `Operator` + `Const`) | Rebuild expression DAGs from flat instructions |
| Value numbering (local) | `BlockValueNumbers::compute` | Per-block hash-consing |
| Value numbering (global) | `ValueNumbering::compute`, `ValueNumberInfo` (associated `Operator`) | Dominator-scoped GVN over any operation identity |
| Redundancy counting | `ValueNumbering::redundant_count` | From GVN results |
| Explicit alias sets | `AliasSets::new` / `merge`; `MemoryAlias` trait | Caller-populated union-find classes usable directly as the alias oracle for memory SSA; an unmerged pair is a proof of disjointness |
| Index-path aliasing | `index_paths_may_overlap` | The `base[i][j]` kernel for alias oracles over indexed storage: unequal arity or unknown components stay conservative, fully known disagreement proves disjointness |
| Exclusive handler extents | `recover_exclusive_extents[_with]`, `promote_exclusive_extents` | Evidence-reporting sibling of `promote_handler_extents`: blocks exclusively reachable from one handler entry along normal edges, boundary and ambiguity issues, and atomic per-region promotion that refuses overlap |
| Dead code analysis | `DeadCode::compute` | Liveness-dead instructions (effect-guarded) and unreachable blocks, reported without mutating — the analysis `dead_code_elimination` applies |
| Purity classification | `cfg_purity`, `block_purity`, `EffectInfo` (associated `Effect`) | Consumer effect vocabularies — machine memory/IO, allocation, panics |
| Metrics | `GraphMetrics::compute` (any rooted view); `CfgMetrics::compute` | Node/edge counts, cyclomatic complexity, nesting depth, instruction density |
| Pattern detection | `detect_patterns` (any view), `detect_cfg_patterns` (adds trampolines + arm orientation) | Diamond, chain, self-loop, empty trampoline |
| Profiling | `CfgProfile::from_edge_weights`, `set_uniform_edge_weights` | Edge-weight-based hot/cold block analysis |
| Tail call detection | `detect_tail_calls` (heuristic), `detect_explicit_tail_calls` (`CallInfo` markers) | |
| Switch table recovery | `detect_switch_tables` (`SwitchSource`), `recover_switch_tables` | Consumer-typed targets: addresses, syntax nodes; dispatch → structured switch |

### Transforms

`PassPipeline` composes heterogeneous named passes over any caller-owned target.
`pass_fn` adapts a closure; implementing `Pass` supports stateful reusable
passes. Schedules retain insertion order and report the completed prefix when a
fallible pass stops execution.

| Transform | Function | Description |
|---|---|---|
| Simplify (all-in-one) | `simplify`, `simplify_mapped` | Unreachable removal + block merging + empty bypass until stable; mapped form composes identity changes |
| Remove unreachable | `remove_unreachable` | DFS reachability pruning |
| Merge blocks | `merge_blocks` | Coalesce single-succ/single-pred chains |
| Remove empty blocks | `remove_empty_blocks` | Bypass empty fallthrough blocks |
| Critical edge splitting | `split_critical_edges`, `split_critical_edges_with` | Insert blocks on multi-succ → multi-pred edges while retaining the original edge identity/payload and mapping both halves |
| Dead code elimination | `dead_code_elimination` (instructions), `remove_dead_code[_mapped]` (plus the structure left dead) | Liveness-based; requires `EffectInfo` so side-effecting code is never silently deleted |
| Edge contraction | `contract_edge`, `contract_edge_mapped` | Merge two blocks connected by a single edge; mapped form preserves surviving edge identities/payloads |
| Node splitting | `split_node`, `split_node_at_points` | Split at one or several validated consumer-selected instruction boundaries |
| Loop rotation | `rotate_loop` | Top-tested → bottom-tested loop form |
| Loop invariant detection | `find_loop_invariants` | Identify hoistable instructions |
| Partial redundancy elimination | `PreAnalysis::compute`, `eliminate_pre` | GVN-based PRE |
| Graph coloring | `interference_graph`, `color_graph` | Interference builder uses `DirectedGraph`; coloring accepts any graph view |
| Linearization | `linearize`, `Emitter` trait, `BlockOrder` | Re-serialize CFG to a flat stream; emitters speak `BlockId`, naming is theirs |
| Width relaxation | `relax_layout` | The assembler fixed point for symbolic layouts: monotone branch widening against caller-resolved labels, offset-dependent (alignment) widths, converged offsets plus the final label context |

### AST recovery (`ir::ast`)

`ir::ast` structures control flow without changing the semantic level of its
generic instruction payload.

| Feature | Description |
|---|---|
| `lift()` → `AstNode<I>` | Recover structured control flow from a CFG |
| `lift_borrowed()` → `AstNode<&I>` | Zero-copy recovery for consumers that keep the source CFG alive; owns only the control tree and reference vectors |
| `lift_with_report()` | The same tree plus a `LiftReport`: every emitted goto with its reason, swept blocks, unstructured regions, unresolved labels — per-construct degradation instead of guessing from the tree |
| `lift_borrowed_with_report()` | The zero-copy tree plus the same complete degradation report |
| `lift_predicated()` | Additionally regionize `Predicated` instruction runs into `Guarded` nodes (ARM IT, GPU wavefront, CMOV) |
| If/then/else | Diamond and triangle patterns; arms stop exactly at the post-dominator merge |
| Loops | `LoopKind` classifies pre-tested (`While` with condition witness and polarity), post-tested (`DoWhile` with latch), and endless loops from natural-loop membership |
| Break/continue | Derived from loop follows and continue points — including machine-shaped CFGs — with labeled multi-level forms wrapping the target loop in a `Label` |
| Switch/case | Case arms carry their dispatch `EdgeId`s (case keys stay on caller edge payloads) and the explicit default arm is captured |
| Try/catch/finally | From region metadata; unknown or malformed handler extents degrade to explicit `Goto`/`Label` flow — reachable code is never dropped |
| Label/goto | Exact post-pass labeling: only blocks a goto actually targets are wrapped |
| Traversal | `child_bodies` (every nested sequence in execution order), `visit`, `for_each_instruction`, and `map_instructions` — the re-leveling hook from opaque payloads to another representation |
| Pseudocode | `to_pseudocode` via `DisplayInstr` — rendering never requires flow classification |

## Extension contracts

The generic graph has no consumer trait requirement when it owns the storage. `DirectedGraph`, `Cfg`, and `KeyedGraph` expose both node and edge views directly. Implement `DenseNodeId` and the three required `DirectedGraphView` methods (`node_count`, `successors`, and `predecessors`) only when adapting another graph store; implement `EdgeGraphView` when algorithms must observe stable edge identities or payloads; add `RootedGraphView` (or use `Rooted`) for entry-requiring algorithms. Instruction traits are opt-in according to which CFG and dataflow features an adapter needs — every associated type is the consumer's own:

```text
DirectedGraph<N, E>       (owned arbitrary graph; no adapter trait)
DirectedGraphView         (existing consumer-owned graph storage)
├── EdgeGraphView         (stable edge identity, endpoints, data, adjacency)
└── RootedGraphView       (adds a distinguished entry node; `Rooted` adapts)

FlowControl               (required only by CfgBuilder)
└── JumpTargets           (goto/label wiring — associated Target)

InstrInfo<Variable = V>   (optional — native IR variables, defs/uses)
├── EffectInfo            (purity, DCE — associated Effect)
├── Predicated            (guarded execution — lift_predicated)
├── CopySource            (copy propagation)
├── ConstantFolder        (constant propagation, SCCP — associated Const)
├── ExprInstr             (expression trees — associated Operator, Const)
├── ValueNumberInfo       (value numbering, PRE — associated Operator)
└── MemoryEventInfo       (locations, address uses, access kinds, and fences — associated Location, Fence)

MemoryAlias<Location>     (optional may-alias oracle consumed by MemorySSA)

DisplayInstr              (rendering only — DOT, pseudocode)
CallInfo                  (call graphs, explicit tail calls — associated Callee)
SwitchSource              (switch table recovery — associated Target)
```

`MemoryEventInfo` is the single instruction-side source of truth for memory.
Directional `MemoryAccess::read`, `write`, and `read_modify_write`
constructors keep loaded definitions, stored uses, and address uses distinct
(`MemoryAccess::opaque` covers table-driven kinds with unknown values);
events also retain exact locations, atomicity, and ordered fences. `MemorySSA`
in `dataflow::memory` merges locations through the caller-provided
`MemoryAlias` relation, then exposes loop-correct reaching definitions,
clobbered states, users, and transitive readers. `MemoryValueFlow` combines it
with the matching ordinary `SsaForm`, making
`value -> store -> memory state -> load -> value` explicit. Returning `false`
from an alias oracle promises the locations cannot overlap at runtime;
`ConservativeMemoryAlias` is the safe fallback when that cannot be proven.

## Workspace

| Crate | Description |
|---|---|
| **cfglib** | Generic graph, CFG, dialect-driven RTL/MLIL/HLIL, SSA, and dataflow framework |

Adapters live with the language they adapt, next to their decoders and
test corpora: the SM4/SM5 decompiler — whose `cfg` module also exposes a
raw CFG + component-granular SSA adapter — is in
[d3dasm](https://github.com/coconutbird/d3dasm); the Java/DEX decompiler
is in cafe. A language picks its entry level: machine-shaped languages
build `ir::rtl` functions (typed parallel transfers over raw storage,
lifted into typed MLIL variables through per-lane SSA webs), while
variable-shaped languages build MLIL directly. cfglib deliberately
defines no generic LLIL — low-level IRs are language-owned, and the
minimal traits they share (`FlowControl`, `InstrInfo`, `Cfg`, `SsaForm`)
are their common surface.

## Adapting a language, IR, or existing graph

For symbol, reference, value-flow, type-relation, import, or grammar graphs, store domain objects directly in `DirectedGraph<N, E>`. Projects that already own adjacency lists can instead implement `DirectedGraphView` for their store, then use traversal, shortest-path, topological-sort, SCC, dominance, loop-detection, metrics, and pattern algorithms without migrating their data. Dense `u32` and `usize` handles work directly; custom handles implement `DenseNodeId`.

For a control-flow and SSA adapter:

1. Build `Cfg<I>` directly with `new_block()` / `add_edge()`, or use `Cfg<I, E>::with_edge_payload()` plus `add_edge_with_payload()` when branch labels, handler order, continuation/call-site identity, or source provenance must survive. Structured unit-payload streams can instead implement `FlowControl` and use `CfgBuilder::build()`.
2. Optionally implement `InstrInfo` with a native `Variable` identity (and its sub-traits) for dataflow analyses.
3. Implement `DisplayInstr` when you want DOT or pseudocode output.

Existing `Cfg<I>` callers remain source-compatible: `E` defaults to `()`, and the original constructors and transform entry points remain. Payload-aware frontends opt into the new methods. `RewriteMap` uses a missing entry for an unchanged identity, an empty replacement list for removal, one replacement for retention/redirect/merge, and several ordered replacements for a split. Stable parallel `EdgeId`s plus caller payloads are also the generic continuation/call-site mechanism; cfglib does not impose a VM-specific continuation type.

The variable type only needs `Clone + Ord` — never `Copy`, never numeric. It can be an architecture enum such as `Register(Rax)` / `Flag(Zero)`, a shader structure such as `(register file, index, component)`, an interned source symbol, or an existing IR value handle. Adapters decide the atomic aliasing unit; for overlapping resources such as x86 subregisters, expose canonical units or every affected unit.

```rust
use cfglib::{Cfg, DominatorTree, InstrInfo, SsaForm};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum X86Variable { Register(u8), Flag(u8), StackSlot(i32) }

struct X86Instruction {
    uses: Vec<X86Variable>,
    defs: Vec<X86Variable>,
}

impl InstrInfo for X86Instruction {
    type Variable = X86Variable;

    fn uses(&self) -> &[Self::Variable] { &self.uses }
    fn defs(&self) -> &[Self::Variable] { &self.defs }
}

let cfg = Cfg::<X86Instruction>::new();
let dominators = DominatorTree::compute(&cfg);
let ssa = SsaForm::compute(&cfg, &dominators);
```

`SsaForm<V>` is a non-mutating view over the source CFG. It stores renamed phi results, operands, instruction uses, and instruction definitions as `SsaValue<V>`, while each `SsaInstruction` keeps a `ProgramPoint` back to the native instruction. Version `0` denotes a live-in or otherwise undefined incoming value.

The concrete shader-bytecode adapter (the d3dasm decompiler's `cfg`
module) derives native register-component identities from decoded masks
and swizzles, retains relative index expressions, classifies multi-result
and UAV read-modify-write operations, and reports observable shader effects
through its own `Sm4Effect` vocabulary.

The `tests/source-cfg.rs` integration test is the executable specification of
the source-language side: interned symbol variables, string/bool constants
through constant propagation, enum operators in expression trees, goto wiring
by label token, and switch recovery over syntax-node targets.

## Development

Install the git hooks once with `prek install` (or `pre-commit install`); both
tools read the same `.pre-commit-config.yaml`. Hygiene checks, the repository
policy script, `cargo fmt`, and pedantic Clippy run at commit time; the full
test and documentation suites run at push time. CI runs the same gates plus an
MSRV (1.85) check and a `no_std` target build. The complete gate list and the
workspace's naming and layout conventions live in `AGENTS.md`.

## License

MIT
