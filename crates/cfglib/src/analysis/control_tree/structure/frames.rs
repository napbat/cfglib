//! The explicit call stack of the structurer.
//!
//! Structuring nests as deeply as the input does, one construct inside
//! another, and the input is untrusted, so the nesting must not use the call
//! stack. Each construct is a [`Frame`]: the walk of a region, a conditional,
//! a switch and its arms, a loop, a `try` and its handlers. A frame holds
//! what its function would have kept in locals. [`Structurer::step`] runs a
//! frame until it needs a nested construct, and answers [`Next::Call`] with
//! the frame for it; [`Structurer::run_frame`] keeps the frames in a vector
//! and resumes the one below with the [`Returned`] value of the one that
//! finished. Each step stores into the arena in the order of its layout (a
//! node's children before the node), whatever the order of the frames.

extern crate alloc;
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::rc::Rc;
use alloc::vec::Vec;

use super::super::node::ControlNode;
use super::Structurer;
use super::branches::{Conditional, Switch, SwitchArm};
use super::loops::{HeaderBody, LoopBuild};
use super::try_catch::{HandlerArms, Handlers, TryBuild};
use super::walk::RegionWalk;
use crate::block::BlockId;

/// The blocks a walk may emit: the enclosing structural bound, or `None` for
/// every block. Frames share a bound, so it is reference counted.
pub(super) type Bound = Option<Rc<BTreeSet<BlockId>>>;

/// One construct being structured, with the state of its walk.
pub(super) enum Frame<'cfg> {
    Region(RegionWalk),
    Conditional(Conditional),
    Switch(Box<Switch>),
    SwitchArm(SwitchArm),
    Loop(Box<LoopBuild>),
    HeaderBody(HeaderBody),
    Try(Box<TryBuild<'cfg>>),
    Handlers(Box<Handlers<'cfg>>),
}

/// What a finished frame hands to the frame that called it.
pub(super) enum Returned {
    /// Nothing has returned yet: the frame is new.
    Start,
    /// The statements of a region, a switch arm, or a loop body.
    Nodes(Vec<ControlNode>),
    /// A construct and the block the walk goes on at after it.
    Construct {
        node: ControlNode,
        next: Option<BlockId>,
    },
    /// The arms of a `try`'s handlers.
    Handlers(HandlerArms),
}

impl Returned {
    pub(super) fn into_nodes(self) -> Vec<ControlNode> {
        match self {
            Self::Nodes(nodes) => nodes,
            _ => unreachable!("a region, an arm, or a body returns its statements"),
        }
    }

    pub(super) fn into_construct(self) -> (ControlNode, Option<BlockId>) {
        match self {
            Self::Construct { node, next } => (node, next),
            _ => unreachable!("a construct returns its node and where the walk goes on"),
        }
    }

    pub(super) fn into_handlers(self) -> HandlerArms {
        match self {
            Self::Handlers(arms) => arms,
            _ => unreachable!("the handlers of a try return their arms"),
        }
    }
}

/// What a frame does after a step.
pub(super) enum Next<'cfg> {
    /// Run this frame to its end, then resume the caller with its value.
    Call(Frame<'cfg>),
    /// The frame is done.
    Return(Returned),
}

impl<'cfg, I, E> Structurer<'cfg, I, E> {
    /// Run `first` and every frame it calls, and return what `first` returns.
    pub(super) fn run_frame(&mut self, first: Frame<'cfg>) -> Returned {
        let mut stack = alloc::vec![first];
        let mut returned = Returned::Start;
        while let Some(frame) = stack.last_mut() {
            match self.step(frame, core::mem::replace(&mut returned, Returned::Start)) {
                Next::Call(callee) => stack.push(callee),
                Next::Return(value) => {
                    stack.pop();
                    returned = value;
                }
            }
        }
        returned
    }

    /// Lift the region that starts at `head`: the statements of the walk along
    /// sequential flow, to the end of the region.
    pub(super) fn lift_region(
        &mut self,
        head: BlockId,
        allowed: Bound,
        stop: Option<BlockId>,
    ) -> Vec<ControlNode> {
        self.run_frame(Frame::Region(RegionWalk::new(head, allowed, stop)))
            .into_nodes()
    }

    fn step(&mut self, frame: &mut Frame<'cfg>, returned: Returned) -> Next<'cfg> {
        match frame {
            Frame::Region(walk) => self.step_region(walk, returned),
            Frame::Conditional(conditional) => self.step_conditional(conditional, returned),
            Frame::Switch(switch) => self.step_switch(switch, returned),
            Frame::SwitchArm(arm) => self.step_switch_arm(arm, returned),
            Frame::Loop(build) => self.step_loop(build, returned),
            Frame::HeaderBody(body) => self.step_header_body(body, returned),
            Frame::Try(build) => self.step_try(build, returned),
            Frame::Handlers(handlers) => self.step_handlers(handlers, returned),
        }
    }
}
