//! Block-existence checks shared by the level builders.
//!
//! MLIL and RTL builders enforce the same two block rules with the same
//! rendered messages; each maps the message into its own error type.

extern crate alloc;

use alloc::format;
use alloc::string::String;

use crate::{BlockId, Cfg};

/// Fails when `block` does not exist in `cfg`.
pub(crate) fn check_block<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> Result<(), String> {
    if cfg.contains_block(block) {
        Ok(())
    } else {
        Err(format!(
            "block {block} is outside a {}-block function",
            cfg.block_count()
        ))
    }
}

/// Fails when `block` is missing or is the synthetic root, naming `role`.
pub(crate) fn check_semantic_block<I, E>(
    cfg: &Cfg<I, E>,
    block: BlockId,
    role: &str,
) -> Result<(), String> {
    check_block(cfg, block)?;
    if block == cfg.entry() {
        return Err(format!("{role} {block} is the synthetic root"));
    }
    Ok(())
}
