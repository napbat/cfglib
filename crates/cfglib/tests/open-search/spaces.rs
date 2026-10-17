//! Lazily generated node spaces: no graph exists, only a successor relation over keys.

/// The Collatz step: halve an even number, triple an odd one and add one.
pub(crate) fn collatz_step(node: u64) -> u64 {
    if node.is_multiple_of(2) {
        node / 2
    } else {
        3 * node + 1
    }
}

/// A chase with one successor per node, which walks the Collatz sequence.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the signature is the `FnMut(&N, &mut Vec<N>)` an open walk takes"
)]
pub(crate) fn collatz_successors(node: &u64, out: &mut Vec<u64>) {
    out.push(collatz_step(*node));
}

/// The Collatz sequence from `start` up to and including its first 1.
pub(crate) fn collatz_trajectory(start: u64) -> Vec<u64> {
    let mut trajectory = vec![start];
    let mut node = start;
    while node != 1 {
        node = collatz_step(node);
        trajectory.push(node);
    }
    trajectory
}

/// How many Collatz steps lead from `node` to the first 1.
pub(crate) fn collatz_steps(node: u64) -> usize {
    collatz_trajectory(node).len() - 1
}

/// The preimages of `node` under the Collatz step: twice the node, and the odd `m` with
/// `3m + 1 == node` when there is one. The space is infinite and never materialized.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the signature is the `FnMut(&N, &mut Vec<N>)` an open walk takes"
)]
pub(crate) fn collatz_predecessors(node: &u64, out: &mut Vec<u64>) {
    out.push(2 * node);
    let below = node - 1;
    if below.is_multiple_of(3) && (below / 3) % 2 == 1 {
        out.push(below / 3);
    }
}

/// The side of the square grid: coordinates run over `0..=GRID_SIDE`.
pub(crate) const GRID_SIDE: u32 = 6;

/// Steps right or down, inside the grid.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the signature is the `FnMut(&N, &mut Vec<N>)` an open walk takes"
)]
pub(crate) fn grid_successors(node: &(u32, u32), out: &mut Vec<(u32, u32)>) {
    let &(x, y) = node;
    if x < GRID_SIDE {
        out.push((x + 1, y));
    }
    if y < GRID_SIDE {
        out.push((x, y + 1));
    }
}

/// `n` choose `k`, as the exact multiplicative product.
pub(crate) fn binomial(n: u64, k: u64) -> u64 {
    (1..=k).fold(1, |product, step| product * (n - k + step) / step)
}
