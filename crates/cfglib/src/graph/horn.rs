//! Horn-clause (AND-OR) derivability over dense fact identities.
//!
//! [`HornClauses`] answers *which facts are derivable* from rules of the form
//! `head <- body1 & body2 & ...`. It is the conjunctive generalisation of
//! graph reachability: an ordinary edge has one source, a clause has a **set**
//! of sources that must **all** hold. Several closures in code intelligence
//! have exactly that shape — grammar nullability (a symbol is nullable when
//! some production has all its symbols nullable), "all arguments constant"
//! propagation, "all callers dead" elimination, and dependency readiness.
//!
//! This deliberately does not ride
//! [`DirectedGraphView`](super::view::DirectedGraphView): a single meet
//! operator cannot be conjunction at the clause class and disjunction at the
//! fact class, so encoding a clause set as a plain graph would either lose the
//! AND or lose the OR. The clause set is its own storage, and the solve is the
//! counter-based Kahn saturation over it (least fixpoint, linear in the total
//! body size).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// A set of Horn clauses `head <- body1 & body2 & ...` over dense fact ids.
///
/// Facts are `0..fact_count`, chosen by the consumer (an interned symbol id,
/// a grammar nonterminal, a node index). Clauses are added in any order; the
/// solved answer is a least fixpoint and therefore independent of insertion
/// order.
///
/// # Examples
///
/// ```
/// use cfglib::HornClauses;
///
/// // Grammar nullability over four symbols: 0 = S, 1 = A, 2 = B, 3 = C,
/// // with the productions `S -> A B`, `A -> ε`, `B -> A A`, `C -> "x"`.
/// let mut clauses = HornClauses::new(4);
/// clauses.add_clause(0, &[1, 2]); // S is nullable when A and B both are
/// clauses.add_clause(1, &[]);     // A -> ε is an axiom
/// clauses.add_clause(2, &[1, 1]); // B is nullable when A is
///                                 // C has no nullable production at all
///
/// assert_eq!(clauses.derivable(), vec![true, true, true, false]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HornClauses {
    /// Number of distinct facts; every head and body id is below it.
    fact_count: usize,
    /// Per clause, the fact it derives.
    heads: Vec<usize>,
    /// Per clause, how many body occurrences it has (duplicates counted).
    body_lens: Vec<usize>,
    /// Per fact, the clauses whose body mentions it — one entry per
    /// occurrence, so a fact repeated in one body decrements it twice.
    dependents: Vec<Vec<usize>>,
}

impl HornClauses {
    /// Create an empty clause set over `fact_count` facts.
    #[must_use]
    pub fn new(fact_count: usize) -> Self {
        Self {
            fact_count,
            heads: Vec::new(),
            body_lens: Vec::new(),
            dependents: vec![Vec::new(); fact_count],
        }
    }

    /// Add the clause `head <- all of body`.
    ///
    /// An empty body makes `head` an axiom. A fact repeated in one body is
    /// counted once **per occurrence**: `a <- b & b` needs `b` derived just
    /// like `a <- b` does, since the same derivation discharges both
    /// occurrences. Adding the same clause twice adds two clauses, which is
    /// harmless — either can derive the head.
    ///
    /// # Panics
    ///
    /// Panics when `head` or any body fact is not below the fact count.
    pub fn add_clause(&mut self, head: usize, body: &[usize]) {
        assert!(head < self.fact_count, "head fact is out of range");
        assert!(
            body.iter().all(|&fact| fact < self.fact_count),
            "body fact is out of range"
        );

        let clause = self.heads.len();
        for &fact in body {
            self.dependents[fact].push(clause);
        }
        self.heads.push(head);
        self.body_lens.push(body.len());
    }

    /// Solve to the least fixpoint: which facts are derivable.
    ///
    /// Returns a dense `Vec<bool>` indexed by fact id. A fact is `true` when
    /// some clause with it at head has every body fact derivable, transitively
    /// from the axioms. Cyclic support derives nothing on its own — `a <- b`
    /// with `b <- a` and no axiom leaves both `false`, which is what a least
    /// fixpoint means.
    ///
    /// The input is untouched, so a clause set can be solved repeatedly (for
    /// example after adding more clauses).
    #[must_use]
    pub fn derivable(&self) -> Vec<bool> {
        let mut derived = vec![false; self.fact_count];
        let mut remaining = self.body_lens.clone();
        // Facts derived but not yet propagated. LIFO: the order in which a
        // fixpoint is reached is immaterial to the fixpoint itself.
        let mut worklist: Vec<usize> = Vec::new();

        // Axioms seed the saturation.
        for (clause, &count) in remaining.iter().enumerate() {
            if count == 0 {
                derive(self.heads[clause], &mut derived, &mut worklist);
            }
        }

        while let Some(fact) = worklist.pop() {
            for &clause in &self.dependents[fact] {
                // Never underflows: a fact enters the worklist at most once,
                // so a clause is decremented at most once per body occurrence.
                let count = &mut remaining[clause];
                *count -= 1;
                if *count == 0 {
                    derive(self.heads[clause], &mut derived, &mut worklist);
                }
            }
        }

        derived
    }
}

/// Mark `fact` derived, enqueueing it the first time.
fn derive(fact: usize, derived: &mut [bool], worklist: &mut Vec<usize>) {
    if !derived[fact] {
        derived[fact] = true;
        worklist.push(fact);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axiom_chains_saturate() {
        // a; b <- a; c <- b.
        let mut clauses = HornClauses::new(3);
        clauses.add_clause(0, &[]);
        clauses.add_clause(1, &[0]);
        clauses.add_clause(2, &[1]);
        assert_eq!(clauses.derivable(), vec![true, true, true]);

        // Solving does not consume or mutate the input.
        assert_eq!(clauses.derivable(), vec![true, true, true]);
    }

    #[test]
    fn a_conjunction_needs_every_body_fact() {
        // a; c <- a & b, with b underivable.
        let mut clauses = HornClauses::new(3);
        clauses.add_clause(0, &[]);
        clauses.add_clause(2, &[0, 1]);
        assert_eq!(clauses.derivable(), vec![true, false, false]);

        // Supplying the missing conjunct unblocks the head.
        clauses.add_clause(1, &[]);
        assert_eq!(clauses.derivable(), vec![true, true, true]);
    }

    #[test]
    fn alternative_clauses_are_a_disjunction() {
        // c <- a (blocked) or c <- b (satisfied): some clause suffices.
        let mut clauses = HornClauses::new(3);
        clauses.add_clause(1, &[]);
        clauses.add_clause(2, &[0]);
        clauses.add_clause(2, &[1]);
        assert_eq!(clauses.derivable(), vec![false, true, true]);
    }

    #[test]
    fn cycles_stay_underivable_without_an_axiom() {
        // a <- b; b <- a: mutual support, no ground.
        let mut clauses = HornClauses::new(2);
        clauses.add_clause(0, &[1]);
        clauses.add_clause(1, &[0]);
        assert_eq!(clauses.derivable(), vec![false, false]);

        // Grounding either one derives both.
        clauses.add_clause(0, &[]);
        assert_eq!(clauses.derivable(), vec![true, true]);
    }

    #[test]
    fn a_self_dependent_clause_alone_derives_nothing() {
        let mut clauses = HornClauses::new(1);
        clauses.add_clause(0, &[0]);
        assert_eq!(clauses.derivable(), vec![false]);
    }

    #[test]
    fn duplicate_body_occurrences_are_discharged_together() {
        // a <- b & b, with b an axiom: one derivation of b discharges both
        // occurrences.
        let mut clauses = HornClauses::new(2);
        clauses.add_clause(1, &[]);
        clauses.add_clause(0, &[1, 1]);
        assert_eq!(clauses.derivable(), vec![true, true]);

        // The same body without the axiom stays blocked.
        let mut blocked = HornClauses::new(2);
        blocked.add_clause(0, &[1, 1]);
        assert_eq!(blocked.derivable(), vec![false, false]);
    }

    #[test]
    fn the_answer_is_order_free() {
        // Grammar nullability: 0 = S, 1 = A, 2 = B, 3 = C.
        // S <- A & B; A <- ; B <- A & A; C <- S & 4-that-never-holds.
        let rules: [(usize, &[usize]); 5] = [
            (0, &[1, 2]),
            (1, &[]),
            (2, &[1, 1]),
            (3, &[0, 4]),
            (4, &[3]),
        ];
        let expected = vec![true, true, true, false, false];

        // Every rotation of the insertion order yields the identical vector.
        for rotation in 0..rules.len() {
            let mut clauses = HornClauses::new(5);
            for offset in 0..rules.len() {
                let (head, body) = rules[(rotation + offset) % rules.len()];
                clauses.add_clause(head, body);
            }
            assert_eq!(clauses.derivable(), expected, "rotation {rotation}");
        }
    }

    #[test]
    fn an_empty_fact_space_is_legal() {
        let clauses = HornClauses::new(0);
        assert_eq!(clauses.derivable().len(), 0);
    }

    #[test]
    #[should_panic(expected = "head fact is out of range")]
    fn an_out_of_range_head_panics() {
        HornClauses::new(1).add_clause(1, &[]);
    }

    #[test]
    #[should_panic(expected = "body fact is out of range")]
    fn an_out_of_range_body_fact_panics() {
        HornClauses::new(1).add_clause(0, &[1]);
    }
}
