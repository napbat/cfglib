//! The one seeded generator of the integration tests and benches. Each
//! constructor names the algorithm a suite was first seeded with, so a seed
//! keeps producing the graph it always produced.

/// A seeded generator, so a failing input reproduces from its seed without a
/// dependency.
pub struct Rng {
    state: u64,
    algorithm: Algorithm,
}

/// How a generator steps its state and turns it into a draw.
enum Algorithm {
    /// `xorshift64*`: three xorshifts, then a multiply of the output.
    XorshiftStar,
    /// `xorshift64`: three xorshifts, and the state is the draw.
    Xorshift,
    /// `splitmix64`: a Weyl sequence through a mixing function.
    SplitMix,
    /// A 64-bit LCG whose draw is the state shifted right by `output_shift`.
    Lcg { output_shift: u32 },
}

impl Rng {
    /// A `xorshift64*` generator. The seed is spread over the state, so seed
    /// zero varies too.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            algorithm: Algorithm::XorshiftStar,
        }
    }

    /// A `xorshift64` generator that starts from `state` as given. A zero
    /// state draws only zeros.
    #[must_use]
    pub const fn xorshift(state: u64) -> Self {
        Self {
            state,
            algorithm: Algorithm::Xorshift,
        }
    }

    /// A `splitmix64` generator that starts from `state` as given.
    #[must_use]
    pub const fn splitmix(state: u64) -> Self {
        Self {
            state,
            algorithm: Algorithm::SplitMix,
        }
    }

    /// A linear congruential generator that starts from `state` as given.
    /// Each draw is the new state shifted right by `output_shift`, which
    /// drops the weak low bits.
    #[must_use]
    pub const fn lcg(state: u64, output_shift: u32) -> Self {
        Self {
            state,
            algorithm: Algorithm::Lcg { output_shift },
        }
    }

    /// The next draw.
    pub fn next_u64(&mut self) -> u64 {
        match self.algorithm {
            Algorithm::XorshiftStar => {
                let mut state = self.state;
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                self.state = state;
                state.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            Algorithm::Xorshift => {
                let mut state = self.state;
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                self.state = state;
                state
            }
            Algorithm::SplitMix => {
                self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut mixed = self.state;
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                mixed ^ (mixed >> 31)
            }
            Algorithm::Lcg { output_shift } => {
                self.state = self
                    .state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                self.state >> output_shift
            }
        }
    }

    /// A value in `0..bound`: the next draw modulo `bound`.
    ///
    /// # Panics
    ///
    /// Panics when `bound` is zero.
    pub fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("a bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below the bound fits in usize")
    }

    /// Whether a draw succeeds with the given percent chance.
    pub fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    /// Puts `items` in a uniformly random order.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for at in (1..items.len()).rev() {
            items.swap(at, self.below(at + 1));
        }
    }
}

#[cfg(test)]
#[test]
fn every_generator_keeps_its_sequence() {
    let first_three = |mut rng: Rng| [rng.next_u64(), rng.next_u64(), rng.next_u64()];
    assert_eq!(
        first_three(Rng::new(7)),
        [
            0x671E_DB9D_E9E9_AFB7,
            0x67E5_3669_D5FB_6B34,
            0x974E_A030_4616_1D80,
        ]
    );
    assert_eq!(
        first_three(Rng::xorshift(0x9E37_79B9_7F4A_7C15)),
        [
            0xDC1B_77AE_0BF3_4DAD,
            0x64F0_EEB9_026E_6076,
            0x7B07_CE91_E590_6136,
        ]
    );
    assert_eq!(
        first_three(Rng::splitmix(1)),
        [
            0x910A_2DEC_8902_5CC1,
            0xBEEB_8DA1_658E_EC67,
            0xF893_A2EE_FB32_555E,
        ]
    );
    assert_eq!(
        first_three(Rng::lcg(7, 33)),
        [0x3F21_945E, 0x7A53_0D3F, 0x740A_AD31]
    );
    assert_eq!(
        first_three(Rng::lcg(0xD1B5_4A32_D192_ED03, 32)),
        [0x0D2B_DACD, 0x2833_35D7, 0x167F_142E]
    );

    assert_eq!(Rng::xorshift(0x2545_F491_4F6C_DD1D).below(1_000), 951);
    let mut rng = Rng::new(7);
    assert!(rng.chance(4));
    assert!(!rng.chance(4));
    let mut items = [0, 1, 2, 3, 4, 5, 6, 7];
    Rng::xorshift(0x9E37_79B9_7F4A_7C15).shuffle(&mut items);
    assert_eq!(items, [2, 1, 3, 6, 7, 0, 4, 5]);
}
