//! Horn-clause (AND-OR) derivability over dense fact identities.
//!
//! [`HornClauses`] answers *which facts are derivable* from rules of the form
//! `head <- body1 & body2 & ...`. It is the conjunctive generalization of
//! graph reachability: an ordinary edge has one source, a clause has a **set**
//! of sources that must **all** hold. Several closures in code intelligence
//! have exactly that shape — grammar nullability (a symbol is nullable when
//! some production has all its symbols nullable), "all arguments constant"
//! propagation, "all callers dead" elimination, and dependency readiness.
//!
//! This deliberately does not ride
//! [`GraphView`](super::view::GraphView): a single meet
//! operator cannot be conjunction at the clause class and disjunction at the
//! fact class, so encoding a clause set as a plain graph would either lose the
//! AND or lose the OR. The clause set is its own storage, and the solve is the
//! counter-based Kahn saturation over it (least fixpoint, linear in the total
//! body size).
//!
//! [`HornSaturation`] keeps those counters between solves, for a clause set
//! that grows in batches: each [`update`](HornSaturation::update) folds in
//! only the clauses added since the last one, so every batch together costs
//! what one solve of the final set does.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::dataflow::bits::DenseBits;

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
#[cfg_attr(feature = "serde", serde(try_from = "stored::Stored"))]
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
    /// Every clause body in clause order, each sorted ascending: the
    /// transpose of `dependents`, kept so that a [`HornSaturation`] reads a
    /// new clause's body in time proportional to its length. It is a function
    /// of `dependents`, so it is not written and is rebuilt on load.
    #[cfg_attr(feature = "serde", serde(skip))]
    bodies: Vec<usize>,
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
            bodies: Vec::new(),
        }
    }

    /// Grow the fact universe to `0..fact_count`, keeping every clause.
    ///
    /// The new facts are underivable until a clause derives them. A count at
    /// or below the current one changes nothing.
    pub fn grow_facts(&mut self, fact_count: usize) {
        if fact_count > self.fact_count {
            self.dependents.resize(fact_count, Vec::new());
            self.fact_count = fact_count;
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
        let start = self.bodies.len();
        self.bodies.extend_from_slice(body);
        self.bodies[start..].sort_unstable();
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
    /// example after adding more clauses). Each call starts over and costs the
    /// whole set; a consumer that adds clauses in batches and asks after each
    /// one should keep a [`saturate`](Self::saturate) state instead, which
    /// costs each batch only its own size.
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

    /// Start an incremental saturation of the clauses so far.
    ///
    /// The state answers what [`derivable`](Self::derivable) does, and keeps
    /// its counters so that [`HornSaturation::update`] after more
    /// [`add_clause`](Self::add_clause) or [`grow_facts`](Self::grow_facts)
    /// calls folds in only the new clauses.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::HornClauses;
    ///
    /// // Dependency readiness as modules load: 0 = app, 1 = net, 2 = tls.
    /// let mut ready = HornClauses::new(3);
    /// ready.add_clause(0, &[1, 2]); // app is ready when net and tls are
    /// let mut state = ready.saturate();
    /// assert!(!state.is_derived(0));
    ///
    /// ready.add_clause(1, &[]); // net loads
    /// assert_eq!(state.update(&ready), [1]);
    ///
    /// ready.grow_facts(4); // 3 = crypto, a new module
    /// ready.add_clause(2, &[3]);
    /// ready.add_clause(3, &[]);
    /// assert_eq!(state.update(&ready), [0, 2, 3]);
    /// assert_eq!(state.derived().ones().collect::<Vec<_>>(), [0, 1, 2, 3]);
    /// ```
    #[must_use]
    pub fn saturate(&self) -> HornSaturation {
        let mut saturation = HornSaturation {
            derived: DenseBits::new(self.fact_count),
            remaining: Vec::with_capacity(self.heads.len()),
            solved_clauses: 0,
            solved_occurrences: 0,
            #[cfg(test)]
            decrements: 0,
        };
        saturation.update(self);
        saturation
    }
}

/// The derived facts of a growing [`HornClauses`], kept between solves.
///
/// Made by [`HornClauses::saturate`]. Each clause keeps the number of its body
/// occurrences that are not derived yet, so a body occurrence is discharged
/// at most once over the life of the state: every
/// [`update`](Self::update) together costs O(S + F) for a final set with
/// total body size S and F facts, where calling
/// [`derivable`](HornClauses::derivable) after each of B batches costs
/// O(B · (S + F)). The answer is the least fixpoint, so it does not depend
/// on how the clauses were split into batches.
///
/// Clauses and facts can only be added. Retracting either would need support
/// counting, which this state does not keep.
#[derive(Debug, Clone)]
pub struct HornSaturation {
    /// The derived facts, over the fact universe of the last update.
    derived: DenseBits,
    /// Per folded clause, its body occurrences whose fact is not derived.
    remaining: Vec<usize>,
    /// How many clauses of the set are folded in.
    solved_clauses: usize,
    /// The total body length of the folded clauses: where the first unfolded
    /// clause's body starts in `HornClauses::bodies`.
    solved_occurrences: usize,
    /// Counters discharged so far, which the cost test reads.
    #[cfg(test)]
    decrements: usize,
}

impl HornSaturation {
    /// Fold in every clause and fact added to `clauses` since the last call,
    /// and propagate; returns the newly derived facts in ascending order.
    ///
    /// `clauses` must be the set this state was made from, grown only by
    /// [`HornClauses::add_clause`] and [`HornClauses::grow_facts`]. The cost
    /// is the total body size of the new clauses plus the body occurrences
    /// the new derivations discharge, each of which is discharged once.
    ///
    /// # Panics
    ///
    /// Panics when `clauses` has fewer clauses or facts than this state has
    /// already folded in, which means it is not the set the state was made
    /// from.
    pub fn update(&mut self, clauses: &HornClauses) -> Vec<usize> {
        let clause_count = clauses.heads.len();
        assert!(
            self.solved_clauses <= clause_count && self.derived.len() <= clauses.fact_count,
            "the clause set is smaller than the one this saturation folded in"
        );
        self.derived.grow(clauses.fact_count);

        // Count every new body against the facts derived before this call and
        // only then derive: a fact derived during the call is propagated once
        // to every clause that counted it, the new ones included.
        for clause in self.solved_clauses..clause_count {
            let end = self.solved_occurrences + clauses.body_lens[clause];
            let body = &clauses.bodies[self.solved_occurrences..end];
            self.solved_occurrences = end;
            let pending = body.iter().filter(|&&fact| !self.derived.get(fact)).count();
            self.remaining.push(pending);
        }
        // The newly derived facts, which double as a first-in first-out
        // worklist: each is pushed once, when it is first derived.
        let mut derived = Vec::new();
        for clause in self.solved_clauses..clause_count {
            if self.remaining[clause] == 0 && self.derived.insert(clauses.heads[clause]) {
                derived.push(clauses.heads[clause]);
            }
        }
        self.solved_clauses = clause_count;

        let mut next = 0;
        while let Some(&fact) = derived.get(next) {
            next += 1;
            for &clause in &clauses.dependents[fact] {
                // Never underflows: the clause counted this occurrence when it
                // was folded in, since the fact was not derived then, and the
                // fact is propagated only once.
                let count = &mut self.remaining[clause];
                *count -= 1;
                #[cfg(test)]
                {
                    self.decrements += 1;
                }
                if *count == 0 && self.derived.insert(clauses.heads[clause]) {
                    derived.push(clauses.heads[clause]);
                }
            }
        }
        derived.sort_unstable();
        derived
    }

    /// Whether `fact` is derived.
    ///
    /// # Panics
    ///
    /// Panics when `fact` is not below the fact count of the last update.
    #[must_use]
    pub fn is_derived(&self, fact: usize) -> bool {
        self.derived.get(fact)
    }

    /// Every derived fact, over the fact universe of the last update.
    #[must_use]
    pub fn derived(&self) -> &DenseBits {
        &self.derived
    }
}

/// The tables of a [`HornClauses`], as they are written, checked on the way in.
///
/// [`HornClauses::derivable`] indexes by every head and decrements a clause
/// once per body occurrence a fact lists, so a document whose tables disagree
/// would panic or underflow there instead of at load.
#[cfg(feature = "serde")]
mod stored {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::HornClauses;

    #[derive(serde::Deserialize)]
    pub struct Stored {
        fact_count: usize,
        heads: Vec<usize>,
        body_lens: Vec<usize>,
        dependents: Vec<Vec<usize>>,
    }

    impl TryFrom<Stored> for HornClauses {
        type Error = &'static str;

        /// Accepts tables that `add_clause` could have produced: one length
        /// per clause, every head a fact, one dependents row per fact, and
        /// each clause listed exactly as often as its body is long.
        fn try_from(stored: Stored) -> Result<Self, Self::Error> {
            let Stored {
                fact_count,
                heads,
                body_lens,
                dependents,
            } = stored;
            if heads.len() != body_lens.len() {
                return Err("the clause tables hold different numbers of clauses");
            }
            if dependents.len() != fact_count {
                return Err("the dependents table does not have one row per fact");
            }
            if heads.iter().any(|&head| head >= fact_count) {
                return Err("a clause head is not a fact");
            }
            let mut mentions = vec![0_usize; heads.len()];
            for &clause in dependents.iter().flatten() {
                *mentions
                    .get_mut(clause)
                    .ok_or("a dependents row names a clause that does not exist")? += 1;
            }
            if mentions != body_lens {
                return Err("a clause body disagrees with the facts that list it");
            }
            // Rebuild each body in clause order: walking the facts ascending
            // leaves every body sorted, as `add_clause` keeps it.
            let mut cursor = mentions;
            let mut total = 0;
            for (start, &len) in cursor.iter_mut().zip(&body_lens) {
                *start = total;
                total += len;
            }
            let mut bodies = vec![0; total];
            for (fact, row) in dependents.iter().enumerate() {
                for &clause in row {
                    bodies[cursor[clause]] = fact;
                    cursor[clause] += 1;
                }
            }
            Ok(Self {
                fact_count,
                heads,
                body_lens,
                dependents,
                bodies,
            })
        }
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
        assert_eq!(clauses.derivable(), [] as [bool; 0]);
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

    /// The facts `derivable` reports, as ascending ids.
    fn derivable_ids(clauses: &HornClauses) -> Vec<usize> {
        let derivable = clauses.derivable();
        (0..derivable.len())
            .filter(|&fact| derivable[fact])
            .collect()
    }

    #[test]
    fn saturation_after_every_random_batch_equals_a_solve_from_scratch() {
        use crate::test_util::rng::Rng;

        let mut grew = 0;
        for seed in 0..60 {
            let mut rng = Rng::new(seed);
            let mut fact_count = 1 + rng.below(6);
            let mut clauses = HornClauses::new(fact_count);
            let mut state = clauses.saturate();
            let mut before = Vec::new();
            for batch in 0..8 {
                if rng.one_in(4) {
                    fact_count += 1 + rng.below(3);
                    clauses.grow_facts(fact_count);
                    grew += 1;
                }
                for _ in 0..rng.below(5) {
                    let head = rng.below(fact_count);
                    let body: Vec<usize> =
                        (0..rng.below(4)).map(|_| rng.below(fact_count)).collect();
                    clauses.add_clause(head, &body);
                }
                let newly = state.update(&clauses);
                let after = derivable_ids(&clauses);
                assert_eq!(
                    state.derived().ones().collect::<Vec<_>>(),
                    after,
                    "seed {seed}, batch {batch}"
                );
                assert_eq!(state.derived().len(), fact_count);
                let expected_new: Vec<usize> = after
                    .iter()
                    .copied()
                    .filter(|fact| !before.contains(fact))
                    .collect();
                assert_eq!(newly, expected_new, "seed {seed}, batch {batch}");
                before = after;
            }
        }
        assert!(grew > 20, "only {grew} batches grew the fact space");
    }

    #[test]
    fn a_batch_may_derive_a_fact_that_a_body_of_the_same_batch_mentions() {
        // Both orders of `x <- ; y <- x & x` in one batch: the body counted
        // against the facts before the batch is discharged by propagation.
        for (first, second) in [((0, &[][..]), (1, &[0, 0][..])), ((1, &[0, 0]), (0, &[]))] {
            let mut clauses = HornClauses::new(2);
            let mut state = clauses.saturate();
            clauses.add_clause(first.0, first.1);
            clauses.add_clause(second.0, second.1);
            assert_eq!(state.update(&clauses), [0, 1]);
            assert_eq!(state.decrements, 2);
        }
    }

    #[test]
    fn a_late_clause_whose_body_is_derived_fires_at_once() {
        let mut clauses = HornClauses::new(3);
        clauses.add_clause(0, &[]);
        let mut state = clauses.saturate();
        assert!(state.is_derived(0));

        // Both occurrences were derived before the clause arrived, so nothing
        // is left to discharge.
        clauses.add_clause(1, &[0, 0]);
        assert_eq!(state.update(&clauses), [1]);
        assert_eq!(state.decrements, 0);
        assert_eq!(
            state.update(&clauses),
            [] as [usize; 0],
            "nothing new, nothing derived"
        );

        // A late clause that supports only itself derives nothing.
        clauses.add_clause(2, &[2]);
        assert_eq!(state.update(&clauses), [] as [usize; 0]);
        assert!(!state.is_derived(2));
    }

    #[test]
    fn a_cycle_is_grounded_by_an_axiom_that_arrives_later() {
        let mut clauses = HornClauses::new(3);
        clauses.add_clause(0, &[1]);
        clauses.add_clause(1, &[0, 2]);
        let mut state = clauses.saturate();
        assert_eq!(state.derived().count_ones(), 0);
        clauses.add_clause(2, &[]);
        assert_eq!(
            state.update(&clauses),
            [2],
            "the cycle still waits on itself"
        );
        clauses.add_clause(0, &[]);
        assert_eq!(state.update(&clauses), [0, 1]);
        assert_eq!(derivable_ids(&clauses), [0, 1, 2]);
    }

    #[test]
    fn growing_the_facts_keeps_the_clauses_and_never_shrinks() {
        let mut clauses = HornClauses::new(1);
        clauses.add_clause(0, &[]);
        let mut state = clauses.saturate();
        clauses.grow_facts(3);
        clauses.grow_facts(2);
        clauses.add_clause(2, &[0]);
        assert_eq!(clauses.derivable(), [true, false, true]);
        assert_eq!(state.update(&clauses), [2]);
        assert_eq!(state.derived().len(), 3);
        assert!(!state.is_derived(1));
    }

    #[test]
    fn batched_saturation_discharges_each_body_occurrence_once() {
        // A chain of 10^5 clauses grounded by one axiom, then 10^4 batches of
        // one clause each: a clause that waits on a new fact, then the axiom
        // that grounds it. `derivable` after every batch would discharge the
        // whole chain every time; the saturation discharges each occurrence
        // once over its whole life.
        const CHAIN: usize = 100_000;
        const BATCHES: usize = 10_000;
        let mut clauses = HornClauses::new(CHAIN);
        clauses.add_clause(0, &[]);
        for fact in 1..CHAIN {
            clauses.add_clause(fact, &[fact - 1]);
        }
        let mut state = clauses.saturate();
        assert_eq!(state.decrements, CHAIN - 1);

        let mut total_body = CHAIN - 1;
        let mut rescan = 0;
        for batch in (0..BATCHES).step_by(2) {
            let waiting = CHAIN + batch;
            clauses.grow_facts(waiting + 2);
            clauses.add_clause(waiting, &[waiting + 1, CHAIN - 1]);
            assert_eq!(state.update(&clauses), [] as [usize; 0]);
            clauses.add_clause(waiting + 1, &[]);
            assert_eq!(state.update(&clauses), [waiting, waiting + 1]);
            total_body += 2;
            // What `derivable` would discharge on each of the two batches:
            // every occurrence of a derived fact, the chain's included.
            rescan += (CHAIN - 1 + batch + 1) + (CHAIN - 1 + batch + 2);
        }
        assert_eq!(state.decrements, CHAIN - 1 + BATCHES / 2);
        assert!(state.decrements <= total_body);
        // Solving from scratch after each batch costs about B times as much.
        assert!(rescan >= state.decrements * (BATCHES * 9 / 10));
        assert_eq!(
            state.derived().count_ones(),
            clauses.derivable().iter().filter(|&&yes| yes).count()
        );
    }

    #[test]
    #[should_panic(expected = "smaller than the one this saturation folded in")]
    fn updating_from_a_smaller_clause_set_panics() {
        let mut clauses = HornClauses::new(2);
        clauses.add_clause(0, &[]);
        let mut state = clauses.saturate();
        state.update(&HornClauses::new(2));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_stored_clause_set_loads_only_when_its_tables_agree() {
        let mut clauses = HornClauses::new(3);
        clauses.add_clause(0, &[]);
        clauses.add_clause(2, &[0, 1, 1]);
        let sound = serde_json::to_value(&clauses).expect("serializes");
        let loaded: HornClauses = serde_json::from_value(sound.clone()).expect("loads");
        assert_eq!(loaded, clauses);
        assert_eq!(loaded.derivable(), clauses.derivable());

        let tamperings = [
            ("a head past the facts", "/heads/0", serde_json::json!(7)),
            (
                "fewer lengths than clauses",
                "/body_lens",
                serde_json::json!([0]),
            ),
            (
                "a row missing for a fact",
                "/dependents",
                serde_json::json!([[], []]),
            ),
            (
                "a row naming a missing clause",
                "/dependents/0",
                serde_json::json!([5]),
            ),
            (
                "a body longer than its listings",
                "/body_lens/1",
                serde_json::json!(4),
            ),
        ];
        for (label, pointer, value) in tamperings {
            let mut document = sound.clone();
            *document
                .pointer_mut(pointer)
                .expect("the encoding has that table") = value;
            assert!(
                serde_json::from_value::<HornClauses>(document).is_err(),
                "{label} must not load"
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_loaded_clause_set_writes_the_same_document_and_keeps_saturating() {
        let mut clauses = HornClauses::new(4);
        clauses.add_clause(3, &[2, 0, 2]);
        clauses.add_clause(0, &[]);
        let document = serde_json::to_value(&clauses).expect("serializes");
        assert_eq!(
            document,
            serde_json::json!({
                "fact_count": 4,
                "heads": [3, 0],
                "body_lens": [3, 0],
                "dependents": [[0], [], [0, 0], []],
            }),
            "the rebuilt bodies are not part of the document"
        );
        let mut loaded: HornClauses = serde_json::from_value(document).expect("loads");
        assert_eq!(loaded, clauses);

        let mut state = loaded.saturate();
        assert_eq!(state.derived().ones().collect::<Vec<_>>(), [0]);
        loaded.add_clause(2, &[0]);
        assert_eq!(state.update(&loaded), [2, 3]);
    }
}
