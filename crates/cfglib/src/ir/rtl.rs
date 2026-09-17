//! Generic register-transfer storage and its typed lifting into MLIL.
//!
//! RTL sits between a language-specific low-level IR and MLIL. Languages
//! whose native form is machine-shaped — raw storage locations written
//! lane-by-lane, untyped bit patterns, parallel multi-destination
//! instructions — express each native instruction here as one
//! [`Statement`]: a group of *parallel* typed transfers whose reads all
//! observe the pre-statement state. That covers shader registers, JVM
//! locals and stack slots, Dalvik registers, machine registers and flags
//! (x86, ARM), and wasm value slots alike; languages that are already
//! variable-shaped skip RTL and build MLIL directly.
//!
//! cfglib deliberately does not define a generic LLIL: low-level IRs are
//! language-owned (a decoded shader program, JVM bytecode, a machine
//! instruction stream), and the existing minimal traits —
//! [`FlowControl`](crate::FlowControl), [`InstrInfo`](crate::InstrInfo),
//! [`Cfg`](crate::Cfg), [`SsaForm`](crate::SsaForm) — are their shared
//! surface. RTL is where those language forms converge: encoding,
//! decoding, and instruction selection stay outside, and every axis —
//! storage, operators, effects, edges, constraints — is dialect-typed.
//!
//! [`lift`] converts an RTL function into an MLIL builder by computing
//! per-lane SSA, uniting versions into def-use *webs* (**live** φ
//! families and co-written lanes — a dead header φ never fuses the
//! lifetimes on either side of a full overwrite), inferring one lane
//! constraint per web through the dialect's [`Constraint`] domain, and
//! emitting MLIL through [`Lift::emit`] — one instruction for
//! storage-flavored dialects ([`Emission::single`]), several for
//! semantic expansion. Edges lift afterwards through [`Lift::lift_edge`]
//! with emitted instruction identities in hand, so exceptional payloads
//! can carry exact throw sites; [`Lifting`] returns the complete
//! [`LiftMaps`] for signatures, exception regions, and provenance.
//! [`lower`] is the downward direction: MLIL onto target RTL under a
//! dialect's [`Lower`] storage and legalization policy, with the full
//! rewrite map in [`Lowered`].
//!
//! Implicit channels — invocation results, delivered exception values,
//! multi-value returns — are ordinary transfers and returns over
//! dialect-chosen storage; live-in webs
//! ([`WebInfo::live_in`](WebInfo::live_in)) enumerate parameters and
//! input channels for signature assignment.
//!
//! Webs make storage reuse harmless: one register reused for a float and
//! then a counter becomes two typed variables, and a parallel transfer's
//! reads never see its own writes because they reference prior versions.
//! The MLIL operations the lift emits are identity-free templates: web
//! variables live only in each instruction's positional `uses`/`defs`
//! lists (reads pair with uses in pre-order), so generic MLIL
//! transformations that rewrite operands stay sound over lifted
//! functions.

mod dialect;
mod emission;
mod error;
mod expr;
mod function;
mod lift;
mod lower;
mod pseudocode;
mod render;
mod statement;
mod template;
mod types;

pub use dialect::{Dialect, Edge, Lift, MlilBridge};
pub use emission::{EdgeContext, Emission, LiftMaps, Lifting};
pub use error::{Error, Result};
pub use expr::{Expr, Place};
pub use function::{Function, FunctionBuilder, ProvenanceMap, Signature};
pub use lift::lift;
pub use lower::{
    Lower, LowerContext, LowerEdgeContext, Lowered, Placement, lower, plan_whole_places,
};
pub use render::{ReadResolver, ResolvedRead, Webs, referenced_webs};
pub use statement::{Lane, Statement, StatementId, StatementNode};
pub use template::{LiftedStatement, VarExpr, WebInfo};
pub use types::{Constraint, Inference, ScalarInference, ScalarType, Shape, ValueShape};

#[cfg(test)]
mod tests;
