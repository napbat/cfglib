//! Depth-first event walk — the fold-shaped counterpart — over an open node space.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::search::{SearchConfig, SearchOrder, Visit, VisitedPolicy, may_expand};

use super::super::marks::Marks;

/// An event of an [`open_depth_first_events`] walk.
///
/// Nodes are reported by reference: the walk owns its frontier, and a node in
/// an open space is usually a compound key — a `(file, name)` pair, a resolved
/// handle — that a consumer should clone only when it keeps one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenDfsEvent<'a, N> {
    /// The node was entered for the first time — under
    /// [`VisitedPolicy::Path`], for the first time *on this path* — at this
    /// depth in hops from its seed.
    ///
    /// This is the only event whose [`Visit`] verdict is read.
    Discover(&'a N, usize),
    /// Every successor of the node has finished, so its subtree is complete.
    ///
    /// Exactly one `Finish` follows every
    /// [`Discover`](OpenDfsEvent::Discover) that the walk does not break out
    /// from under, and the two nest strictly, so a visitor-side stack pushed
    /// on discovery is reduced here.
    Finish(&'a N),
    /// The walk declined to enter the node, at the depth it was reached from:
    /// its mark was already set, so it produces no
    /// [`Discover`](OpenDfsEvent::Discover), no
    /// [`Finish`](OpenDfsEvent::Finish), and no successors.
    ///
    /// Under [`VisitedPolicy::Path`] the node is already on the current path,
    /// which is the cycle guard — a consumer diagnosing a cyclic inheritance
    /// or import graph reports it from here. Under [`VisitedPolicy::Global`]
    /// it was entered earlier in the walk, and the event is the fold
    /// contribution this route does *not* get.
    ///
    /// The two are one event deliberately: distinguishing "on the path" from
    /// "already finished" under `Global` needs the tri-color state a dense
    /// [`depth_first_events`](crate::graph::search::depth_first_events) keeps, and a
    /// visitor that wants it already has the gray interval — it is exactly
    /// between a node's `Discover` and its `Finish`.
    Refused(&'a N, usize),
}

/// One frame of the explicit stack an [`open_depth_first_events`] walk keeps.
///
/// The successors themselves live in the walk's single arena, not in a `Vec`
/// per frame: frames are pushed and popped strictly last in, first out, so the
/// top frame always owns the arena's tail and a pop truncates back to where
/// that frame's successors began. One allocation for the walk (two with the
/// scratch buffer the closure is handed), instead of one per entered node.
struct OpenDfsFrame<N> {
    /// The node the frame stands on.
    node: N,
    /// Its depth in hops from the seed.
    depth: usize,
    /// Where its successors start in the arena, as the closure pushed them;
    /// the region is empty when the node was pruned or sat at the depth bound.
    start: usize,
    /// The next successor to enter, as an arena index. The frame's successors
    /// are exhausted when it reaches the arena's length, since the top frame's
    /// region runs to the end.
    cursor: usize,
}

/// Walk a lazily discovered node space depth-first from `seeds`, reporting
/// every discovery, every finish, and every refused re-entry, and returning
/// the value the callback broke with (or `None` when the walk ran to
/// completion).
///
/// [`crate::open_search`] answers questions about nodes and can only stop; this
/// answers questions that are *folds*, where a node's answer is computed from
/// its subtrees' answers and therefore only exists on the unwind. C++ member
/// lookup is the motivating shape: a class answers with its own declaration
/// if it has one, else with the answer of exactly one yielding base
/// subobject, and two yielding subobjects make the name ambiguous. Under
/// [`VisitedPolicy::Path`] the marks release at
/// [`Finish`](OpenDfsEvent::Finish), so a diamond's shared base is folded
/// once per route through it — and that re-fold is precisely what makes the
/// ambiguity visible. A globally marked walk answers the same question
/// "unambiguous", confidently and wrongly.
///
/// `successors` is handed a node and a scratch buffer, **cleared before every
/// call**, and pushes that node's successors *in the order they should be
/// explored*, exactly as in [`crate::open_search`]. Nodes are compared and marked by
/// `Ord`.
///
/// # Events
///
/// The event order is deterministic and part of the API. The walk is
/// iterative — an explicit frame stack, no recursion — so a deep space costs
/// heap, not stack.
///
/// - **[`Discover`](OpenDfsEvent::Discover)`(node, depth)`** is emitted when
///   the walk enters a node, seeds first at depth 0 in the order given and
///   then successors in the order the closure pushed them.
/// - **[`Finish`](OpenDfsEvent::Finish)`(node)`** is emitted when the node's
///   last successor has finished, strictly last in, first out. Every
///   `Discover` is matched by exactly one `Finish`, the walk breaking being
///   the one exception (below).
/// - **[`Refused`](OpenDfsEvent::Refused)`(node, depth)`** replaces that pair
///   when the mark policy declines the re-entry. The re-entry is *reported*
///   rather than silently dropped, because for a fold it is a real answer:
///   a cycle under [`VisitedPolicy::Path`], a lost contribution under
///   [`VisitedPolicy::Global`].
/// - **[`Visit::Skip`]** prunes: the successor closure is not called for that
///   node — which matters when discovering successors costs a file read —
///   and the node **still finishes**, immediately, as a leaf. Pruning is a
///   statement about the subtree, not about the node, and a fold that had to
///   reduce a pruned node somewhere other than its `Finish` would need two
///   reduction sites for one answer. (This is the C++ rule itself: a class
///   that declares the name hides every base declaration, so its bases are
///   never searched, and its own declaration is the answer it finishes with.)
/// - The verdict returned for a `Finish` or a `Refused` is **ignored**; there
///   is nothing left to prune. Return
///   `ControlFlow::Continue(`[`Visit::Descend`]`)`.
/// - **`config.max_depth` bounds expansion, not visiting**: a node at the
///   bound is discovered and finished, and its successors are never
///   discovered. `config.visited` chooses the marks. The walk is depth-first
///   by construction, because a [`Finish`](OpenDfsEvent::Finish) *is* an
///   unwind.
/// - **Seeds**: under [`VisitedPolicy::Global`] a seed already reached from
///   an earlier seed is `Refused` at depth 0; under [`VisitedPolicy::Path`]
///   the stack is empty between seeds, so every mark has been released and a
///   repeated seed is walked again.
/// - **[`ControlFlow::Break`]** returns immediately with `Some(value)`: the
///   frames still on the stack do **not** finish, so a visitor that breaks
///   owns whatever it broke with rather than a completed fold.
///
/// # Examples
///
/// The C++ ambiguity, folded on the unwind — a visitor-side stack pushed at
/// `Discover` and reduced at `Finish`:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{OpenDfsEvent, SearchConfig, Visit, VisitedPolicy, open_depth_first_events};
///
/// // The classic non-virtual diamond: `d` derives from `b1` and `b2`, both
/// // derive from `base`, and only `base` declares the member.
/// let bases = |class: &&'static str, out: &mut Vec<&'static str>| match *class {
///     "d" => out.extend(["b1", "b2"]),
///     "b1" | "b2" => out.push("base"),
///     _ => {}
/// };
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// enum Lookup {
///     Nothing,
///     Found(&'static str),
///     Ambiguous,
/// }
///
/// // Exactly one yielding base subobject answers; two are an ambiguity.
/// let merge = |answer: Lookup, from_base: Lookup| match (answer, from_base) {
///     (Lookup::Nothing, other) | (other, Lookup::Nothing) => other,
///     _ => Lookup::Ambiguous,
/// };
///
/// let mut frames: Vec<Lookup> = Vec::new();
/// let mut answer = Lookup::Nothing;
/// let outcome = open_depth_first_events::<_, ()>(
///     ["d"],
///     SearchConfig { visited: VisitedPolicy::Path, ..SearchConfig::default() },
///     bases,
///     |event| {
///         match event {
///             OpenDfsEvent::Discover(class, _) => {
///                 if *class == "base" {
///                     // A declaration hides the bases below it: prune, and
///                     // finish immediately with this answer.
///                     frames.push(Lookup::Found(*class));
///                     return ControlFlow::Continue(Visit::Skip);
///                 }
///                 frames.push(Lookup::Nothing);
///             }
///             OpenDfsEvent::Finish(_) => {
///                 let found = frames.pop().unwrap_or(Lookup::Nothing);
///                 match frames.last_mut() {
///                     Some(parent) => *parent = merge(*parent, found),
///                     None => answer = found,
///                 }
///             }
///             OpenDfsEvent::Refused(..) => {}
///         }
///         ControlFlow::Continue(Visit::Descend)
///     },
/// );
///
/// assert_eq!(outcome, None);
/// // `base` was folded once per route, which is what makes it ambiguous.
/// assert_eq!(answer, Lookup::Ambiguous);
/// ```
///
/// # Panics
///
/// Panics when `config.order` is [`SearchOrder::BreadthFirst`]: a
/// breadth-first frontier never unwinds, so it has no finish to report.
#[must_use]
pub fn open_depth_first_events<N: Clone + Ord, B>(
    seeds: impl IntoIterator<Item = N>,
    config: SearchConfig,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut on_event: impl FnMut(OpenDfsEvent<'_, N>) -> ControlFlow<B, Visit>,
) -> Option<B> {
    assert!(
        config.order == SearchOrder::DepthFirst,
        "open_depth_first_events requires SearchOrder::DepthFirst: a breadth-first frontier never unwinds, so a node could never finish"
    );
    let mut marks = Marks::new();
    let mut stack: Vec<OpenDfsFrame<N>> = Vec::new();
    // Every frame's successors, appended as the frame is pushed and truncated
    // away as it pops, plus the buffer the closure writes into — which has to
    // be a separate one, because the closure is promised an empty `Vec` (a
    // consumer that sorts its successors would otherwise sort the arena).
    let mut arena: Vec<N> = Vec::new();
    let mut scratch: Vec<N> = Vec::new();

    // Enter a node: discover it and push its frame, or report the re-entry
    // the mark policy refuses. Used for seeds and successors alike, so the
    // two cannot drift apart.
    macro_rules! enter {
        ($node:expr, $depth:expr) => {{
            let node: N = $node;
            let depth: usize = $depth;
            if marks.insert(node.clone()) {
                let verdict = match on_event(OpenDfsEvent::Discover(&node, depth)) {
                    ControlFlow::Break(value) => return Some(value),
                    ControlFlow::Continue(verdict) => verdict,
                };
                scratch.clear();
                if matches!(verdict, Visit::Descend) && may_expand(config.max_depth, depth) {
                    successors(&node, &mut scratch);
                }
                let start = arena.len();
                arena.append(&mut scratch);
                stack.push(OpenDfsFrame {
                    node,
                    depth,
                    start,
                    cursor: start,
                });
            } else if let ControlFlow::Break(value) = on_event(OpenDfsEvent::Refused(&node, depth))
            {
                return Some(value);
            }
        }};
    }

    for seed in seeds {
        enter!(seed, 0);
        while let Some(frame) = stack.last_mut() {
            if frame.cursor < arena.len() {
                let next = arena[frame.cursor].clone();
                let depth = frame.depth + 1;
                frame.cursor += 1;
                enter!(next, depth);
                continue;
            }
            let Some(finished) = stack.pop() else { break };
            arena.truncate(finished.start);
            if matches!(config.visited, VisitedPolicy::Path) {
                marks.remove(&finished.node);
            }
            if let ControlFlow::Break(value) = on_event(OpenDfsEvent::Finish(&finished.node)) {
                return Some(value);
            }
        }
    }

    None
}
