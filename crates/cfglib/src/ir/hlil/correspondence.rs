//! The table that relates a flat function to its structured form.

extern crate alloc;

use alloc::vec::Vec;

use crate::ProgramPoint;
use crate::graph::view::DenseId;

use super::recover::Recovery;
use super::view::Entity;

/// The tables that relate a flat function to the structured function lifted
/// from it.
///
/// Spans and every other provenance of the flat function belong to the
/// consumer. It composes them with these tables, which name the structured
/// entity that carries each flat instruction and variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Correspondence<S, E, V> {
    /// The flat program point and the structured entity that carries it: a
    /// statement, or an expression when the definition was inlined into its
    /// consumer. A pure control transfer has no row. Sorted by point, with at
    /// most one row per point. Empty when the lift keeps no metadata.
    pub instructions: Vec<(ProgramPoint, Entity<S, E, V>)>,
    /// The structured variable of each flat variable, indexed by the flat
    /// variable's [`DenseId::index`]. A builder that issues handles in call
    /// order gives the identity map.
    pub variables: Vec<V>,
}

impl<S: DenseId, E: DenseId, V: DenseId> Correspondence<S, E, V> {
    /// Moves every row onto the function that `recovery` rebuilt from the
    /// structured function this table describes.
    ///
    /// A statement or expression row follows the entity to its copy, or to the
    /// recovered construct that consumed it. A row whose entity has no copy
    /// is dropped: the expressions of a cleanup that a recovered region
    /// regenerates are examples. A variable row follows the variable to its
    /// copy.
    ///
    /// # Panics
    ///
    /// Panics when a variable of this table is not a variable of the function
    /// that `recovery` rebuilt.
    #[must_use]
    pub fn remap<S2, E2, V2, F>(
        self,
        recovery: &Recovery<F, S2, E2, V2>,
    ) -> Correspondence<S2, E2, V2>
    where
        S2: DenseId,
        E2: DenseId,
        V2: DenseId,
    {
        let instructions = self
            .instructions
            .into_iter()
            .filter_map(|(point, entity)| {
                let entity = match entity {
                    Entity::Statement(statement) => {
                        recovery.statement(statement).map(Entity::Statement)
                    }
                    Entity::Expression(expression) => {
                        recovery.expression(expression).map(Entity::Expression)
                    }
                    Entity::Variable(variable) => {
                        Some(Entity::Variable(recovery.variable(variable)))
                    }
                }?;
                Some((point, entity))
            })
            .collect();
        let variables = self
            .variables
            .into_iter()
            .map(|variable| recovery.variable(variable))
            .collect();
        Correspondence {
            instructions,
            variables,
        }
    }
}
