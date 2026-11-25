//! An infallible problem stated through the fallible traits, as a consumer
//! does to stream it: every operation forwards, and the error is
//! `Infallible`.

use std::convert::Infallible;

use cfglib::{
    EdgeProblem, EdgeRef, EdgeView, GraphView, NodeProblem, Problem, RootedView, TryEdgeProblem,
    TryNodeProblem, TryProblem,
};

/// The fallible form of an infallible problem.
pub struct Lifted<'p, P>(pub &'p P);

impl<V: RootedView, P: Problem<V>> TryProblem<V> for Lifted<'_, P> {
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> cfglib::Direction {
        Problem::direction(self.0)
    }

    fn bottom(&self, view: &V) -> P::Fact {
        Problem::bottom(self.0, view)
    }

    fn boundary(&self, view: &V) -> Result<P::Fact, Infallible> {
        Ok(Problem::boundary(self.0, view))
    }

    fn meet(&self, _: &V, _: V::NodeId, a: &P::Fact, b: &P::Fact) -> Result<P::Fact, Infallible> {
        Ok(Problem::meet(self.0, a, b))
    }

    fn transfer(&self, view: &V, block: V::NodeId, input: &P::Fact) -> Result<P::Fact, Infallible> {
        Ok(Problem::transfer(self.0, view, block, input))
    }
}

impl<G: GraphView, P: NodeProblem<G>> TryNodeProblem<G> for Lifted<'_, P> {
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> cfglib::Direction {
        NodeProblem::direction(self.0)
    }

    fn bottom(&self, graph: &G) -> P::Fact {
        NodeProblem::bottom(self.0, graph)
    }

    fn boundary(&self, graph: &G) -> Result<P::Fact, Infallible> {
        Ok(NodeProblem::boundary(self.0, graph))
    }

    fn meet(&self, _: &G, _: G::NodeId, a: &P::Fact, b: &P::Fact) -> Result<P::Fact, Infallible> {
        Ok(NodeProblem::meet(self.0, a, b))
    }

    fn transfer(&self, graph: &G, node: G::NodeId, input: &P::Fact) -> Result<P::Fact, Infallible> {
        Ok(NodeProblem::transfer(self.0, graph, node, input))
    }
}

impl<G: EdgeView, P: EdgeProblem<G>> TryEdgeProblem<G> for Lifted<'_, P> {
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> cfglib::Direction {
        EdgeProblem::direction(self.0)
    }

    fn bottom(&self, graph: &G) -> P::Fact {
        EdgeProblem::bottom(self.0, graph)
    }

    fn boundary(&self, graph: &G, node: G::NodeId) -> Result<Option<P::Fact>, Infallible> {
        Ok(EdgeProblem::boundary(self.0, graph, node))
    }

    fn meet(&self, _: &G, _: G::NodeId, a: &P::Fact, b: &P::Fact) -> Result<P::Fact, Infallible> {
        Ok(EdgeProblem::meet(self.0, a, b))
    }

    fn transfer_node(
        &self,
        graph: &G,
        node: G::NodeId,
        flow_fact: &P::Fact,
    ) -> Result<P::Fact, Infallible> {
        Ok(EdgeProblem::transfer_node(self.0, graph, node, flow_fact))
    }

    fn transfer_edge(
        &self,
        graph: &G,
        edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
        node_input: &P::Fact,
        node_output: &P::Fact,
    ) -> Result<P::Fact, Infallible> {
        Ok(EdgeProblem::transfer_edge(
            self.0,
            graph,
            edge,
            node_input,
            node_output,
        ))
    }
}
