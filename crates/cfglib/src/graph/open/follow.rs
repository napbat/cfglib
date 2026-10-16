//! Out-degree ≤ 1 chases with hop bounds and cycle guards.

extern crate alloc;

use alloc::vec::Vec;
use smallvec::SmallVec;

/// The nodes a chase has stood on after its seed. The first eight stay inline,
/// which is longer than the alias and re-export chains real chases walk, so the
/// common chase never allocates.
type Hops<N> = SmallVec<[N; 8]>;

/// Follow an out-degree ≤ 1 chase from `seed` and return where it ends.
///
/// `step` reports the next node, or `None` at the end of the chain. The chase
/// stops — returning the **last** node it stood on — when `step` yields
/// `None`, when `max_hops` steps have been taken, or when the next node is
/// already on the path walked so far.
///
/// The cycle guard is the **full path**, not just the seed: an alias chain
/// that re-enters itself three hops in is a cycle even though it never
/// returns to where it started. `max_hops` keeps that scan small (and is the
/// real bound for a chase over an untrusted graph).
///
/// # Examples
///
/// ```
/// use cfglib::follow;
///
/// // A chain of type aliases: 0 -> 1 -> 2, and 2 is the real definition.
/// let chain = |node: &u32| match node {
///     0 => Some(1),
///     1 => Some(2),
///     _ => None,
/// };
/// assert_eq!(follow(0, 16, chain), 2);
///
/// // A cycle stops at the last node before it closes.
/// let cyclic = |node: &u32| Some((node + 1) % 3);
/// assert_eq!(follow(0, 16, cyclic), 2);
///
/// // So does the hop bound.
/// assert_eq!(follow(0, 1, cyclic), 1);
/// ```
#[must_use]
pub fn follow<N: PartialEq + Clone>(
    seed: N,
    max_hops: usize,
    step: impl FnMut(&N) -> Option<N>,
) -> N {
    chase(&seed, max_hops, step).pop().unwrap_or(seed)
}

/// Follow an out-degree ≤ 1 chase from `seed` and return the whole chain,
/// `seed` first.
///
/// The chain is what [`follow`] walks — same termination, same full-path
/// cycle guard — kept instead of discarded, for a consumer that reports the
/// route (a diagnostic naming every alias between a use and its definition)
/// or inspects it. The result always contains `seed`, so it is never empty,
/// and holds at most `max_hops + 1` nodes.
///
/// # Examples
///
/// ```
/// use cfglib::follow_path;
///
/// let cyclic = |node: &u32| Some((node + 1) % 3);
/// assert_eq!(follow_path(0, 16, cyclic), vec![0, 1, 2]);
/// assert_eq!(follow_path(0, 1, cyclic), vec![0, 1]);
///
/// let end = |_: &u32| None;
/// assert_eq!(follow_path(7, 16, end), vec![7]);
/// ```
#[must_use]
pub fn follow_path<N: PartialEq + Clone>(
    seed: N,
    max_hops: usize,
    step: impl FnMut(&N) -> Option<N>,
) -> Vec<N> {
    let hops = chase(&seed, max_hops, step);
    let mut chain = Vec::with_capacity(hops.len() + 1);
    chain.push(seed);
    chain.extend(hops);
    chain
}

/// The chase both public forms run: every node stood on after `seed`, in
/// order. The guard is the whole path, so `seed` counts as stood on.
fn chase<N: PartialEq>(
    seed: &N,
    max_hops: usize,
    mut step: impl FnMut(&N) -> Option<N>,
) -> Hops<N> {
    let mut hops = Hops::new();

    for _ in 0..max_hops {
        let current = hops.last().unwrap_or(seed);
        let Some(next) = step(current) else { break };
        if *seed == next || hops.contains(&next) {
            break;
        }
        hops.push(next);
    }

    hops
}

#[cfg(test)]
mod tests;
