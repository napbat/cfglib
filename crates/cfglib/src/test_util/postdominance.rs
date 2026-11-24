//! Post-dominance for graphs without a unique exit, from its definition by
//! fixpoint, as a test oracle.
//!
//! Hecker, Bischof and Snelting ("On Time-sensitive Control Dependencies",
//! TOPLAS 44(1), 2021, theorem 2.1) define two relations by the rules
//!
//! - (self) `n ⊑ n`;
//! - (succ) if `n` reaches `m` and `m ⊑ x` for every successor `x` of `n`,
//!   then `m ⊑ n`,
//!
//! read as "`m` post-dominates `n`". The greatest fixpoint is sink
//! post-dominance, over the paths that end in a sink component and stay in
//! it; the least is max post-dominance, over every maximal path. Both are
//! computed here pair by pair, with nothing borrowed from the production
//! code. Node `n` reaches itself.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

/// Which fixpoint of the rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fixpoint {
    /// The greatest: sink post-dominance.
    Sink,
    /// The least: max post-dominance.
    Max,
}

/// `reach[n][m]`: a path of any length, zero included, leads from `n` to
/// `m`.
fn closure(successors: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let count = successors.len();
    let mut reach = vec![vec![false; count]; count];
    for (start, row) in reach.iter_mut().enumerate() {
        let mut stack = vec![start];
        row[start] = true;
        while let Some(node) = stack.pop() {
            for &next in &successors[node] {
                if !row[next] {
                    row[next] = true;
                    stack.push(next);
                }
            }
        }
    }
    reach
}

/// `relation[m][n]`: `m` post-dominates `n`, as the fixpoint `which` of the
/// rules over the graph `successors` describes.
pub(crate) fn post_dominance(successors: &[Vec<usize>], which: Fixpoint) -> Vec<Vec<bool>> {
    let count = successors.len();
    let reach = closure(successors);
    let mut relation: Vec<Vec<bool>> = (0..count)
        .map(|m| {
            (0..count)
                .map(|n| m == n || (which == Fixpoint::Sink && reach[n][m]))
                .collect()
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for m in 0..count {
            for n in 0..count {
                if m == n || !reach[n][m] {
                    continue;
                }
                let derived = successors[n].iter().all(|&next| relation[m][next]);
                if relation[m][n] != derived {
                    relation[m][n] = derived;
                    changed = true;
                }
            }
        }
    }
    relation
}

/// The control dependences of the two-successor definition over
/// `relation`: `(x, y)` when `x` has successors `n` and `m` with `y`
/// post-dominating `n` and not `m`.
pub(crate) fn two_successor_dependences(
    successors: &[Vec<usize>],
    relation: &[Vec<bool>],
) -> BTreeSet<(usize, usize)> {
    let mut dependences = BTreeSet::new();
    for (controller, targets) in successors.iter().enumerate() {
        for (dependent, row) in relation.iter().enumerate() {
            let reached = targets.iter().any(|&next| row[next]);
            let missed = targets.iter().any(|&next| !row[next]);
            if reached && missed {
                dependences.insert((controller, dependent));
            }
        }
    }
    dependences
}

/// The subgraph of `successors` that `root` reaches, renumbered densely in
/// ascending order, with the old index of each new node.
pub(crate) fn reachable_part(
    successors: &[Vec<usize>],
    root: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let reach = closure(successors);
    let kept: Vec<usize> = (0..successors.len())
        .filter(|&node| reach[root][node])
        .collect();
    let mut renumber = vec![usize::MAX; successors.len()];
    for (new, &old) in kept.iter().enumerate() {
        renumber[old] = new;
    }
    let part = kept
        .iter()
        .map(|&old| successors[old].iter().map(|&next| renumber[next]).collect())
        .collect();
    (part, kept)
}
