//! Out-of-SSA lowering the way a backend that owns its register machine does
//! it: take the phi copies cfglib computes, materialize them as parallel
//! moves on the edges, and run the result.
//!
//! The oracle is an interpreter: every lowered program must behave exactly
//! like the program it came from. The classic hazards are exercised on
//! purpose. The swap problem needs the copies of one edge to be a parallel
//! assignment, and the lost-copy problem needs the critical edge to be split
//! before a copy is placed on it. Phi webs are checked against an independent
//! union-find over the same phis.

mod copies;
mod hazards;
mod lowering;
mod machine;
mod programs;
mod webs;
