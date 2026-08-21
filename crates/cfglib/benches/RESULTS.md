# Local CFG performance results

These results compare `main` at `5331ad5684d14568ef94a2e8714026dec5712117`
with the frozen optimized code at
`cf0e130a6e179de5f0d97e13079959bd5cae40eb` on 2026-08-21. The report is a
follow-up documentation commit, so the optimized code commit is intentionally
named rather than self-referencing the report commit.

## Method and provenance

- The exact benchmark source and Cargo benchmark target from `cf0e130` were
  overlaid on the baseline worktree; byte-for-byte equality was checked with
  `cmp` before measurement. Library source remained at the named revision in
  each worktree.
- Every selected case first runs a case-specific semantic oracle outside the
  measurement. A failed oracle stops the run; an invalid duration or unmatched
  filter exits with status 2.
- CPU measurements used an AMD RYZEN AI MAX+ 395, logical CPU 2, Linux
  7.0.0-30-generic, Rust 1.96.0 (`ac68faa20`) and LLVM 22.1.2. CPU mode links
  directly to `System`, pins the process with `taskset -c 2`, and reports the
  median of seven samples.
- The full matrices used at least 150 ms per sample. Noise-sensitive rows were
  rerun as isolated, alternating comparisons at 300 ms; the isolated values
  replace their full-matrix values below.
- Allocation mode is a separately compiled binary. It runs one unmeasured
  warm-up and one measured complete operation through a counting wrapper over
  `System`. The columns are successful allocations/reallocations, total
  requested bytes, and incremental peak live requested bytes. They are not
  process RSS.
- Most analysis fixtures contain 4,096 nodes. Construction cases intentionally
  construct inside the measured operation. Mutation cases clone their fixture
  inside the operation and return the complete mutated CFG, so adjacent
  clone-only controls expose setup cost.

See [README.md](README.md) for commands and metric definitions.

## Result summary

The retained changes are strongest where the old implementation repeatedly
rescanned or reallocated whole-graph state:

- reverse-ID interval analysis: 2,947.56x faster and 470,712,140 -> 51,676
  requested bytes;
- dominator depths: 550.97x faster with unchanged allocation pressure;
- interval analysis: 246.11x faster and 539,489 -> 656 allocations;
- SCCP: 211.61x faster;
- redirecting 4,096 incoming edges: 114.76x faster;
- empty-chain removal: 83.38x faster;
- linear-chain merging: 71.14x faster;
- constant propagation: 44.54x faster and 75,479,433 -> 138,305 requested
  bytes;
- control dependence and multi-latch loop detection: 35.46x and 30.09x
  faster, respectively.

Ratios below are `baseline / optimized`; values above 1 are faster. Differences
within a few percent should be treated as noise unless confirmed on target
hardware. Three baseline mutation rows are deliberately marked invalid rather
than assigned a speedup: their semantic oracles expose correctness defects in
`main`.

## Complete CPU matrix

All times are nanoseconds per complete operation.

| Case | `main` | `cf0e130` | Ratio |
|---|---:|---:|---:|
| `cfg_build_branchy` | 169058.9 | 148663.9 | 1.14x |
| `cfg_builder_if_else_chain` | 261342.2 | 242937.7 | 1.08x |
| `cfg_builder_conditional_break_chain` | 398413.0 | 339530.2 | 1.17x |
| `cfg_builder_two_case_switch_chain` | 186824.1 | 190779.8 | 0.98x |
| `cfg_builder_eight_case_switch_chain` | 176400.2 | 176004.1 | 1.00x |
| `directed_build_branchy` | 29023.4 | 29175.5 | 0.99x |
| `cfg_depth_first_preorder` | 17482.8 | 17245.5 | 1.01x |
| `cfg_breadth_first` | 11401.8 | 11168.3 | 1.02x |
| `directed_breadth_first_edges` | 18104.9 | 17123.2 | 1.06x |
| `directed_shortest_path` | 14251.5 | 13294.5 | 1.07x |
| `directed_shortest_path_edges` | 16941.3 | 15832.4 | 1.07x |
| `directed_nearest_common_ancestor` | 36601.2 | 26930.9 | 1.36x |
| `directed_common_ancestors` | 34248.9 | 28429.6 | 1.20x |
| `cfg_dominators` | 99538.6 | 47590.1 | 2.09x |
| `cfg_dominance_frontiers` | 83008.9 | 55036.2 | 1.51x |
| `cfg_post_dominators` | 137027.5 | 95632.6 | 1.43x |
| `cfg_post_dominators_many_exits` | 189711.2 | 166892.4 | 1.14x |
| `cfg_control_dependence_graph` | 2336271.3 | 65878.2 | 35.46x |
| `cfg_dominator_depths_linear` | 2060582.5 | 3739.9 | 550.97x |
| `directed_tarjan_scc` | 124956.8 | 102754.6 | 1.22x |
| `directed_detect_loops_multilatch` | 17289510.9 | 574616.9 | 30.09x |
| `cfg_interval_analysis` | 32177979.6 | 130745.1 | 246.11x |
| `directed_interval_reverse_id_chain` | 331588528.0 | 112495.9 | 2947.56x |
| `directed_node_fixpoint_bool` | 61988.9 | 18380.4 | 3.37x |
| `directed_node_fixpoint_wide` | 563791.9 | 543580.7 | 1.04x |
| `cfg_fixpoint_bool` | 93189.4 | 90205.1 | 1.03x |
| `cfg_fixpoint_wide` | 204141.8 | 204637.4 | 1.00x |
| `cfg_sccp_independent_constants` | 11562111.8 | 54638.4 | 211.61x |
| `cfg_build_ssa_linear` | 1763705.8 | 160890.1 | 10.96x |
| `cfg_place_phis_phi_storm` | 478533.9 | 186064.9 | 2.57x |
| `cfg_build_ssa_phi_storm` | 1262588.4 | 940243.0 | 1.34x |
| `cfg_global_value_numbering_linear` | 1969821.0 | 452728.2 | 4.35x |
| `cfg_constprop_independent_constants` | 4970119.7 | 111582.4 | 44.54x |
| `cfg_clone_linear` | 53923.8 | 54357.9 | 0.99x |
| `cfg_clone_merge_linear` | 6552729.3 | 92104.8 | 71.14x |
| `cfg_clone_empty_chain` | 24301.3 | 24559.6 | 0.99x |
| `cfg_clone_remove_empty_chain` | 6452540.0 | 77386.1 | 83.38x |
| `cfg_clone_high_fan_in` | 47749.5 | 48976.5 | 0.97x |
| `cfg_clone_redirect_high_fan_in` | 5897055.0 | 51387.4 | 114.76x |
| `cfg_clone_weighted_high_fan_out` | 2847.5 | 2864.5 | 0.99x |
| `cfg_clone_split_weighted_high_fan_out` | 8959.8 | 5695.7 | 1.57x |
| `cfg_clone_merge_weighted_high_fan_out` | invalid oracle | 10284.1 | correctness restored |
| `cfg_clone_contract_weighted_high_fan_out` | invalid oracle | 4536.2 | correctness restored |
| `cfg_clone_irreducible_small` | 6515.8 | 6529.4 | 1.00x |
| `cfg_clone_make_reducible_small` | 182641.8 | 22876.3 | 7.98x |
| `cfg_clone_irreducible_large` | 12465.2 | 12503.4 | 1.00x |
| `cfg_clone_make_reducible_large` | 21892891.9 | 9786737.0 | 2.24x |
| `cfg_clone_weighted_irreducible` | 78.0 | 79.4 | 0.98x |
| `cfg_clone_make_reducible_weighted` | invalid oracle | 703.8 | correctness restored |

## Complete allocation matrix

Each cell is `allocations / requested bytes / peak live requested bytes` for
one complete operation.

| Case | `main` | `cf0e130` |
|---|---:|---:|
| `cfg_build_branchy` | 4272 / 1525280 / 804848 | 4272 / 1525280 / 804848 |
| `cfg_builder_if_else_chain` | 8241 / 2440864 / 1229040 | 8241 / 2440864 / 1229040 |
| `cfg_builder_conditional_break_chain` | 6197 / 4783776 / 2392304 | 6197 / 4783776 / 2392304 |
| `cfg_builder_two_case_switch_chain` | 8241 / 2440864 / 1229040 | 8241 / 2440864 / 1229040 |
| `cfg_builder_eight_case_switch_chain` | 7217 / 2461344 / 1249536 | 7217 / 2461344 / 1249536 |
| `directed_build_branchy` | 4 / 411648 / 411648 | 4 / 411648 / 411648 |
| `cfg_depth_first_preorder` | 14 / 36868 / 28688 | 14 / 36868 / 28688 |
| `cfg_breadth_first` | 3 / 20496 / 20496 | 3 / 20496 / 20496 |
| `directed_breadth_first_edges` | 15 / 206989 / 108733 | 14 / 200720 / 102464 |
| `directed_shortest_path` | 15 / 69636 / 53264 | 15 / 53252 / 36880 |
| `directed_shortest_path_edges` | 13 / 53248 / 45072 | 13 / 36864 / 28688 |
| `directed_nearest_common_ancestor` | 26 / 196704 / 147520 | 4 / 65568 / 65552 |
| `directed_common_ancestors` | 37 / 393216 / 245760 | 26 / 294832 / 180224 |
| `cfg_dominators` | 8210 / 417752 / 147456 | 20 / 286712 / 147456 |
| `cfg_dominance_frontiers` | 7286 / 342464 / 276960 | 3191 / 276944 / 276944 |
| `cfg_post_dominators` | 8231 / 914478 / 557172 | 23 / 503858 / 196672 |
| `cfg_post_dominators_many_exits` | 8250 / 978874 / 606276 | 33 / 569322 / 217089 |
| `cfg_control_dependence_graph` | 790 / 435064 / 422872 | 265 / 421904 / 417792 |
| `cfg_dominator_depths_linear` | 1 / 16384 / 16384 | 1 / 16384 / 16384 |
| `directed_tarjan_scc` | 2367 / 735696 / 490160 | 2367 / 735696 / 490160 |
| `directed_detect_loops_multilatch` | 88234 / 6153396 / 53884 | 433 / 56336 / 41036 |
| `cfg_interval_analysis` | 539489 / 30655100 / 1037280 | 656 / 61452 / 56888 |
| `directed_interval_reverse_id_chain` | 8397854 / 470712140 / 1031916 | 685 / 51676 / 51580 |
| `directed_node_fixpoint_bool` | 8321 / 80348 / 28684 | 4 / 28672 / 28672 |
| `directed_node_fixpoint_wide` | 6844 / 9821788 / 4254732 | 4225 / 8698880 / 4254720 |
| `cfg_fixpoint_bool` | 395 / 216672 / 94224 | 392 / 85612 / 48744 |
| `cfg_fixpoint_wide` | 4877 / 9856032 / 4267024 | 4874 / 9823276 / 4255784 |
| `cfg_sccp_independent_constants` | 181 / 98088 / 81768 | 181 / 98088 / 81768 |
| `cfg_build_ssa_linear` | 10948 / 793388 / 450100 | 6188 / 739964 / 461268 |
| `cfg_place_phis_phi_storm` | 6647 / 512400 / 218976 | 5017 / 403096 / 207032 |
| `cfg_build_ssa_phi_storm` | 28475 / 3050972 / 2266740 | 26799 / 2941565 / 2267149 |
| `cfg_global_value_numbering_linear` | 8872 / 1048248 / 1048248 | 6827 / 1048264 / 1048264 |
| `cfg_constprop_independent_constants` | 116719 / 75479433 / 117004 | 690 / 138305 / 111272 |
| `cfg_clone_linear` | 2052 / 303072 / 303072 | 2052 / 303072 / 303072 |
| `cfg_clone_merge_linear` | 14347 / 21479284 / 382928 | 2066 / 329700 / 315360 |
| `cfg_clone_empty_chain` | 6 / 294888 / 294888 | 6 / 294888 / 294888 |
| `cfg_clone_remove_empty_chain` | 10240 / 21305292 / 305148 | 10 / 305148 / 305148 |
| `cfg_clone_high_fan_in` | 5 / 606544 / 606544 | 5 / 606544 / 606544 |
| `cfg_clone_redirect_high_fan_in` | 16 / 655664 / 639312 | 6 / 622928 / 622928 |
| `cfg_clone_weighted_high_fan_out` | 9 / 164220 / 164220 | 9 / 164220 / 164220 |
| `cfg_clone_split_weighted_high_fan_out` | 25 / 476236 / 328428 | 13 / 427100 / 295660 |
| `cfg_clone_merge_weighted_high_fan_out` | invalid oracle | 26 / 213407 / 197003 |
| `cfg_clone_contract_weighted_high_fan_out` | invalid oracle | 11 / 164240 / 164236 |
| `cfg_clone_irreducible_small` | 6 / 98736 / 98736 | 6 / 98736 / 98736 |
| `cfg_clone_make_reducible_small` | 3180 / 725011 / 209904 | 77 / 396245 / 209904 |
| `cfg_clone_irreducible_large` | 6 / 172176 / 172176 | 6 / 172176 / 172176 |
| `cfg_clone_make_reducible_large` | 1337345 / 103608396 / 393504 | 16911 / 61330791 / 393504 |
| `cfg_clone_weighted_irreducible` | 8 / 624 / 624 | 8 / 624 / 624 |
| `cfg_clone_make_reducible_weighted` | invalid oracle | 45 / 2728 / 1428 |

## Correctness and review hardening

The semantic oracles reject three `main` operations before timing:

- weighted high-fanout merge and contract leave adjacency pointing at removed
  edge slots (`edge has been removed`);
- weighted `make_reducible` loses edge weights (`None` instead of the original
  weight).

The optimized rows therefore mean correctness restored, not measured speedup.
Additional review fixes preserve weights, edge order, parallel-edge
multiplicity, entry identity, and panic atomicity; reject invalid
post-dominator exits; support empty interval views and bounded consumer node
IDs; and propagate backward facts through disconnected components.

Red/green evidence was observed, not inferred. A temporary test suite against
the pre-hardening optimization commit `b871606` failed all six targeted cases:
invalid post-dominator exit, empty interval view, invalid redirect mutation,
entry-consuming merge, entry-consuming contract, and reducibility weight
preservation. A seventh focused test showed disconnected backward propagation
failing at `b871606`. The corresponding final tests pass. The three baseline
benchmark oracle failures were also re-probed directly at `5331ad5`.

## Retained implementation changes

- Reused direct adjacency iterators and dense parent/mark tables in dominance,
  traversal, shortest-path, ancestor, and control-dependence analyses.
- Memoized dominator depths in two passes. Whole-tree consumers share compact
  child links and use `u32` depths only when the node count proves the sentinel
  cannot alias a real depth.
- Normalized large post-dominator exit lists once, used binary membership for
  sorted unique lists, preserved duplicate-edge symmetry, and kept the virtual
  exit private as `usize` so bounded consumer IDs never represent it.
- Replaced repeated loop-body reachability searches with one multi-source
  reverse traversal, and rewrote interval discovery around predecessor
  counters and a worklist.
- Used dense worklists and epoch marks for node fixpoints, phi placement, CDG,
  and SSA while retaining consumer-visible edge ordering. The generic CFG
  solver keeps its allocation-ID scheduler after paired wide-fact testing and
  now includes every disconnected component for backward analyses.
- Batched SCCP rescans, maintained constant facts incrementally, and shared
  dominator child traversal in SSA and global value numbering.
- Made block cleanup single-pass and added private bulk incoming/outgoing edge
  moves that preserve edge IDs, adjacency ordering, kinds, and weights.
- Combined irreducibility detection with target discovery and replaced repeated
  predecessor filtering/reachability searches with one partition and bitmap.
- Kept every new helper private; normalized rustdoc JSON has the same 475 public
  paths, kinds, and signatures as `main`.

## Rejected or deferred avenues

- A FIFO CFG-fixpoint worklist made Boolean facts faster but regressed the
  paired wide-fact case from about 204 us to 543 us. Exact-priority heap and
  fact-representation heuristics either retained the regression, added memory,
  or made scheduling depend on an unjustified proxy. They were reverted; the
  settled solver is CPU-neutral on both paired cases and requests fewer bytes.
- `SmallVec` builder arm storage reduced allocations, but an isolated
  direct-`System` run regressed the common if/else fixture by 16.5%; it was
  reverted. An inline break-exit variant was also slower.
- Alternative Tarjan component materialization, exact-priority bitmap/heap
  worklists, and several post-dominator exit representations lost on CPU or
  memory in focused A/B runs and were reverted.
- A cached live-edge counter would add representation and serialization
  invariants for only one production caller, so it was deferred.
- Whole-graph arenas or CSR storage conflict with stable edge IDs, slice
  adjacency, mutable instruction `Vec` access, or serialized shape. Scratch
  local to transformations and analyses was densified instead.
- No hot loop exposed a portable arithmetic kernel where explicit SSE beat
  LLVM. Measured hotspots were allocation, graph traversal, repeated scans,
  and worklist order; LLVM already vectorized the wide fact kernel.
- Advanced controlled node splitting can improve pathological irreducible CFGs
  but adds code growth and minimum-split complexity. The retained splitter is
  2.24x faster on the large adversarial fixture, and published empirical work
  reports irreducibility as uncommon, so the larger algorithm was deferred.

Research consulted: [Tarjan on reducibility](https://doi.org/10.1145/800125.804040),
[Janssen and Corporaal on controlled node splitting](https://www.cs.tufts.edu/comp/150FP/archive/johan-jansson/node-splitting.pdf),
[Cytron et al. on SSA and control dependence](https://rsim.cs.uiuc.edu/arch/qual_papers/compilers/toplas91.pdf),
[IBM on loops, dominators, and frontiers](https://research.ibm.com/publications/on-loops-dominators-and-dominance-frontiers),
and [Stanier on irreducibility in practice](https://doi.org/10.1002/spe.1059).

## Verification

The final tree is checked with:

- stable all-feature and no-default-feature workspace tests;
- declared MSRV Rust 1.85.0 all-feature and no-default-feature workspace tests;
- stable and latest-nightly all-target/all-feature Clippy with warnings and
  `clippy::pedantic` denied;
- rustfmt, rustdoc with warnings denied, and `git diff --check`;
- the allocation-instrumented benchmark target under Clippy;
- normalized nightly rustdoc-JSON public API comparison;
- benchmark fail-closed probes for zero duration and unmatched filters.

The exact final command results are recorded in the draft PR checks after the
documentation commit.

## Limitations and scoped standards exceptions

- These are deterministic synthetic fixtures on one machine, not application
  traces. CPU frequency boost was enabled; pinning, medians, and isolated paired
  reruns reduce noise but do not replace a target-hardware or empirical corpus.
- Linux hardware performance counters were unavailable
  (`perf_event_paranoid=4`); Callgrind informed hotspot discovery, while final
  comparisons use wall-clock time and allocator requests.
- The external standards review covered evidence, red/green verification,
  truthful comments, public contracts, MSRV/no-std compatibility, deduplication,
  and patch hygiene. At the user's direction this PR does not change existing
  CI workflow enforcement or action pinning. Pre-existing decorative comment
  separators outside the patch are also unchanged.
