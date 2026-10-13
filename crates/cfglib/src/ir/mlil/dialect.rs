//! The vocabulary of one medium-level dialect.

use core::fmt::Debug;
use core::hash::Hash;

/// The vocabulary of one medium-level dialect.
///
/// A dialect is a closed set of operations with its own value types and edge
/// metadata. Every associated type is plain data stored by the consumer.
pub trait Dialect: Copy + Eq + Ord + Hash + Debug + 'static {
    /// One operation together with its immediate attributes, such as a
    /// comparison predicate or a memory-access descriptor.
    type Op: Clone + Eq + Hash + Debug;

    /// The type of one value. Every value has one type, fixed when it is
    /// defined.
    type Type: Clone + Eq + Hash + Debug;

    /// The metadata of one successor record, such as the case value of a
    /// switch arm.
    type Edge: Clone + Eq + Debug;
}
