//! A seeded generator for randomized tests.

/// A xorshift64 generator, so a failing input reproduces from its seed
/// without a dependency.
pub(crate) struct Rng(u64);

impl Rng {
    /// A generator whose sequence is a pure function of `seed`, including
    /// seed zero.
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    /// A value in `0..bound`, which must be nonzero.
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below bound fits in usize")
    }

    /// Whether a draw lands in one of `outcomes` equally likely buckets.
    pub(crate) fn one_in(&mut self, outcomes: usize) -> bool {
        self.below(outcomes) == 0
    }

    /// Whether a draw succeeds with the given percent chance.
    pub(crate) fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    /// Put `items` in a uniformly random order.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for end in (1..items.len()).rev() {
            let other = self.below(end + 1);
            items.swap(end, other);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::Rng;

    #[test]
    fn a_seed_fixes_the_whole_sequence() {
        let mut first = Rng::new(7);
        let mut second = Rng::new(7);
        let mut other = Rng::new(8);
        let draws: Vec<usize> = (0..32).map(|_| first.below(1_000)).collect();
        assert_eq!(
            draws,
            (0..32).map(|_| second.below(1_000)).collect::<Vec<_>>()
        );
        assert_ne!(
            draws,
            (0..32).map(|_| other.below(1_000)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn seed_zero_still_varies() {
        let mut rng = Rng::new(0);
        let draws: Vec<usize> = (0..32).map(|_| rng.below(1_000)).collect();
        assert!(draws.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn below_stays_in_range_and_reaches_every_value() {
        let mut rng = Rng::new(1);
        let mut seen = [false; 5];
        for _ in 0..200 {
            seen[rng.below(5)] = true;
        }
        assert!(seen.into_iter().all(core::convert::identity));
    }

    #[test]
    fn chance_and_one_in_respect_their_extremes() {
        let mut rng = Rng::new(2);
        for _ in 0..100 {
            assert!(!rng.chance(0));
            assert!(rng.chance(100));
            assert!(rng.one_in(1));
        }
    }

    #[test]
    fn shuffle_permutes_without_losing_items() {
        let mut rng = Rng::new(3);
        let mut items: Vec<u32> = (0..16).collect();
        rng.shuffle(&mut items);
        assert_ne!(items, (0..16).collect::<Vec<_>>());
        items.sort_unstable();
        assert_eq!(items, (0..16).collect::<Vec<_>>());
    }
}
