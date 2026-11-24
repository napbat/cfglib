//! The path algebra of a forward, union-meet gen/kill bit-vector problem.
//!
//! A gen/kill transfer `x ↦ generate ∪ (x ∖ kill)` stays a gen/kill transfer
//! under composition, union and star, so the paths to a node evaluate to one
//! transfer, and applying it to the root's input gives the meet over all
//! paths. For a distributive problem that is the least fixpoint as well,
//! which is what lets the same answer come from
//! [`solve_node_problem`](crate::solve_node_problem).

use super::PathAlgebra;
use crate::dataflow::bits::DenseBits;

/// The bit-vector transfer `x ↦ generate ∪ (x ∖ kill)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenKill {
    generate: DenseBits,
    kill: DenseBits,
}

impl GenKill {
    /// The transfer that adds `generate` and removes `kill`; an index in
    /// both is generated.
    ///
    /// # Panics
    ///
    /// Panics when the two sets have different universes.
    #[must_use]
    pub fn new(generate: DenseBits, kill: DenseBits) -> Self {
        assert_eq!(
            generate.len(),
            kill.len(),
            "generate and kill sets must share one universe"
        );
        Self { generate, kill }
    }

    /// The transfer that changes nothing, over `universe` indices.
    #[must_use]
    pub fn identity(universe: usize) -> Self {
        Self::new(DenseBits::new(universe), DenseBits::new(universe))
    }

    /// The indices the transfer adds.
    #[must_use]
    pub fn generate(&self) -> &DenseBits {
        &self.generate
    }

    /// The indices the transfer removes unless it also adds them.
    #[must_use]
    pub fn kill(&self) -> &DenseBits {
        &self.kill
    }

    /// The transfer applied to `input`.
    ///
    /// # Panics
    ///
    /// Panics when `input` has a different universe.
    #[must_use]
    pub fn apply(&self, input: &DenseBits) -> DenseBits {
        let mut output = difference(input, &self.kill);
        output.union_with(&self.generate);
        output
    }

    /// This transfer followed by `next`.
    ///
    /// # Panics
    ///
    /// Panics when the two transfers have different universes.
    #[must_use]
    pub fn then(&self, next: &Self) -> Self {
        let mut generate = difference(&self.generate, &next.kill);
        generate.union_with(&next.generate);
        let mut kill = self.kill.clone();
        kill.union_with(&next.kill);
        Self { generate, kill }
    }
}

/// `left ∖ right`.
fn difference(left: &DenseBits, right: &DenseBits) -> DenseBits {
    assert_eq!(left.len(), right.len(), "sets must share one universe");
    let mut result = DenseBits::new(left.len());
    for index in left.ones().filter(|&index| !right.get(index)) {
        result.insert(index);
    }
    result
}

/// The [`PathAlgebra`] of a forward bit-vector problem whose meet is union:
/// each edge carries a [`GenKill`] transfer, and the value of a set of paths
/// is the transfer of their union, or `None` for no path.
///
/// `None` is the zero rather than the transfer to the empty set because a
/// transfer that generates bits would turn "no path" into "some facts" when
/// composed after it.
///
/// # Examples
///
/// ```
/// use cfglib::{DenseBits, GenKill, GenKillAlgebra, Graph, Rooted, interpret, path_expressions};
///
/// // Reaching definitions of one variable: `def0` defines it in the entry,
/// // `def1` in one arm of a branch.
/// let mut graph = Graph::new();
/// let [entry, then, join] = [(); 3].map(|()| graph.add_node(()));
/// graph.add_edge(entry, then, ());
/// graph.add_edge(entry, join, ());
/// graph.add_edge(then, join, ());
/// let expressions = path_expressions(&Rooted::new(&graph, entry));
///
/// let transfer = |node| {
///     let mut generate = DenseBits::new(2);
///     let mut kill = DenseBits::new(2);
///     if node == then {
///         generate.insert(1);
///         kill.insert(0);
///     }
///     GenKill::new(generate, kill)
/// };
/// // An edge carries the transfer of the node it enters.
/// let algebra = GenKillAlgebra::new(2, |edge| transfer(graph.edge(edge).target()));
/// let values = interpret(&expressions, &algebra);
/// let path = values[join.index()].clone().flatten().expect("the entry reaches the join");
///
/// let mut defined = DenseBits::new(2);
/// defined.insert(0); // the entry's own definition
/// assert_eq!(path.apply(&defined).ones().collect::<Vec<_>>(), [0, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct GenKillAlgebra<F> {
    universe: usize,
    transfer: F,
}

impl<F> GenKillAlgebra<F> {
    /// The algebra over `universe` indices, where `transfer` gives each
    /// edge's transfer.
    #[must_use]
    pub const fn new(universe: usize, transfer: F) -> Self {
        Self { universe, transfer }
    }
}

impl<E, F: Fn(E) -> GenKill> PathAlgebra<E> for GenKillAlgebra<F> {
    type Value = Option<GenKill>;

    fn zero(&self) -> Self::Value {
        None
    }

    fn one(&self) -> Self::Value {
        Some(GenKill::identity(self.universe))
    }

    fn edge(&self, edge: E) -> Self::Value {
        Some((self.transfer)(edge))
    }

    /// The union of two transfers: generated by either, killed by both.
    fn add(&self, a: &Self::Value, b: &Self::Value) -> Self::Value {
        match (a, b) {
            (None, other) | (other, None) => other.clone(),
            (Some(a), Some(b)) => {
                let mut generate = a.generate.clone();
                generate.union_with(&b.generate);
                let mut kill = DenseBits::new(a.kill.len());
                for index in a.kill.intersection(&b.kill) {
                    kill.insert(index);
                }
                Some(GenKill { generate, kill })
            }
        }
    }

    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value {
        Some(a.as_ref()?.then(b.as_ref()?))
    }

    /// A transfer composed with itself is itself, so its star is the union
    /// of it and the identity: it generates and kills nothing more.
    fn star(&self, a: &Self::Value) -> Self::Value {
        match a {
            None => self.one(),
            Some(a) => Some(GenKill {
                generate: a.generate.clone(),
                kill: DenseBits::new(a.kill.len()),
            }),
        }
    }
}
