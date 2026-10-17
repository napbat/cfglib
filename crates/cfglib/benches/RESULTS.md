# Local optimization results

Measured on Windows on 2026-08-24, starting from revision `43d3f07`. These are
synthetic regression fixtures, not claims about every consumer workload. This
file is a dated record of that session, not a description of the current tree:
later changes have moved the numbers (the allocation counts of several cases
below are different today), and the harness's `coverage:` line, not this page,
states how many cases and facade functions a run covers. Rerun the commands in
[README.md](README.md) for current figures.

## Screening comparison

The table compares the unmodified starting revision with the optimized working
tree on the same machine. Both used the direct `System` allocator, seven timing
samples, and `CFGLIB_BENCH_MS=1`. That short target is useful for ranking large
regressions while iterating, but the absolute values are intentionally treated
as approximate. Mutation timings include the fixture clone named by each case.

| Case | Before (ns/op) | After (ns/op) | Speedup |
| --- | ---: | ---: | ---: |
| `cfg_dominators` | 222,400 | 49,066 | 4.53x |
| `cfg_control_dependence_graph` | 2,347,700 | 75,800 | 30.97x |
| `directed_detect_loops_multilatch` | 18,842,000 | 578,750 | 32.56x |
| `cfg_interval_analysis` | 45,103,800 | 137,013 | 329.19x |
| `directed_interval_reverse_id_chain` | 851,031,600 | 110,340 | 7,712.81x |
| `directed_node_fixpoint_bool` | 444,833 | 21,611 | 20.58x |
| `cfg_sccp_independent_constants` | 20,204,000 | 53,414 | 378.26x |
| `cfg_build_ssa_linear` | 2,427,100 | 240,663 | 10.09x |
| `cfg_constprop_independent_constants` | 3,752,000 | 72,567 | 51.70x |
| `directed_breadth_first_edges` | 100,560 | 14,963 | 6.72x |
| `directed_shortest_path_edges` | 99,655 | 15,906 | 6.27x |
| `cfg_clone_merge_linear` | 131,924,300 | 110,867 | 1,189.94x |
| `cfg_clone_remove_empty_chain` | 47,257,400 | 84,708 | 557.88x |
| `cfg_clone_redirect_high_fan_in` | 6,900,500 | 82,150 | 84.00x |
| `cfg_clone_split_weighted_high_fan_out` | 407,700 | 7,536 | 54.10x |
| `cfg_clone_merge_weighted_high_fan_out` | 218,995,500 | 12,593 | 17,390.26x |
| `cfg_clone_make_reducible_large` | 44,566,300 | 12,070,300 | 3.69x |

The starting global-value-numbering fixture overflowed the Windows stack on a
2,048-block dominator chain; the iterative implementation completes it. The
starting weighted reducibility transformation also failed its semantic oracle
because copied outgoing edges lost their weights; the optimized version retains
edge kind, weight, identity rules, and adjacency order.

## Verification run

A longer run of the optimized tree with `CFGLIB_BENCH_MS=25` completed every
case and all semantic oracles. The registry verified benchmark coverage for
every root-facade free function. The allocation-instrumented build also
completed every case and verified that each measured operation returned to its
starting live-byte baseline.

Selected results from that facade run are below. CPU values are medians of
seven samples with a 25 ms minimum per sample. Allocation values are one
steady-state operation after warm-up. Mutation cases include fixture cloning.

| Case | CPU (ns/op) | Allocations | Requested bytes |
| --- | ---: | ---: | ---: |
| `api_recover_expressions` | 142,807.1 | 2,529 | 207,104 |
| `api_abstract_interpret` | 7,216.8 | 66 | 7,304 |
| `api_copy_propagation` | 1,221.6 | 35 | 4,593 |
| `api_try_solve_problem` | 13,910.6 | 63 | 23,128 |
| `api_solve_edge_problem` | 35,191.1 | 1,104 | 17,921 |
| `api_search` | 4,120.3 | 11 | 36,808 |
| `api_search_with_scratch` | 3,453.2 | 0 | 0 |
| `api_open_breadth_first_paths` | 231,399.1 | 2,049 | 12,578,960 |
| `api_kosaraju_scc` | 34,976.3 | 623 | 163,856 |
| `api_program_dependence_graph` | 965,648.3 | 1,582 | 410,770 |
| `api_verify_edge_view` | 524,693.6 | 1,310 | 270,388 |
| `api_merge_blocks_mapped` | 52,619.0 | 864 | 77,364 |
| `api_dead_code_elimination` | 78,912.0 | 1,304 | 145,600 |
| `api_interference_graph` | 139,325.0 | 1,229 | 116,192 |
| `api_linearize` | 21,609.6 | 600 | 207,896 |

The zero-allocation reusable-scratch search is a useful harness sanity check:
the corresponding allocating `search` entry point reports 11 allocations on
the same 1,024-node workload. Use the commands in [README.md](README.md) to
reproduce either mode; pinning an otherwise-idle machine and increasing the
target duration is recommended before making small-factor decisions.
