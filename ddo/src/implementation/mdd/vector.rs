//! This is an adaptation of the vector based architecture which implements all
//! the pruning techniques that I have proposed in my PhD thesis (RUB, LocB, EBPO).
//! It also implements the techniques we proposed in
//!
//! ``Decision Diagram-Based Branch-and-Bound with Caching
//! for Dominance and Suboptimality Detection''.

use std::{
    collections::{hash_map::Entry, HashSet},
    fmt::Debug,
    fs,
    hash::{BuildHasherDefault, Hash},
    sync::Arc,
};

use derive_builder::Builder;
// use derive_builder::Builder;
use fxhash::{FxHashMap, FxHashSet};

use crate::{
    CompilationInput, CompilationStrategy, CompilationType, Completion, CutsetType, Decision,
    DecisionDiagram, DominanceCheckResult, NodeFlags, Problem, Reason, Solution, SubProblem,
    FRONTIER, LAST_EXACT_LAYER,
};

/// The identifier of a node: it indicates the position of the referenced node
/// in the ’nodes’ vector of the mdd structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct NodeId(usize, usize);

/// The identifier of an edge: it indicates the position of the referenced edge
/// in the ’edges’ vector of the mdd structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct EdgeId(usize);

/// The identifier of a layer: it indicates the position of the referenced layer
/// in the 'layers' vector of the mdd structure.
#[derive(Debug, Clone, Copy)]
struct LayerId(usize);

/// Represents an effective node from the decision diagram
#[derive(Debug, Clone)]
struct Node<T> {
    /// The length of the longest path between the problem root and this
    /// specific node
    value_top: isize,
    /// The length of the longest path between this node and the terminal node.
    ///
    /// ### Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    value_bot: isize,
    /// The identifier of the last edge on the longest path between the problem
    /// root and this node if it exists.
    best: Option<EdgeId>,
    /// The identifier of the latest edge having been added to the adjacency
    /// list of this node. (Edges, by themselves form a kind of linked structure)
    rub: isize,
    /// A threshold value to be stored in the cache that conditions the
    /// re-exploration of other nodes with the same state.
    ///
    /// ### Note
    /// This field is only ever populated after the MDD has been fully unrolled.
    theta: Option<isize>,
    /// A group of flag telling if the node is an exact node, if it is a relaxed
    /// node (helps to determine if the best path is an exact path) and if the
    /// node is reachable in a backwards traversal of the MDD starting at the
    /// terminal node.
    flags: NodeFlags,
    /// The number of decisions that have been made since the problem root
    depth: usize,
    /// Vector of incoming edges of node
    incoming: Vec<EdgeId>,
    /// Vector of outgoing edges of node
    outgoing: Vec<EdgeId>,
    /// The state associated to this node
    state: Arc<T>,
    /// (Under-)estimate of conflicts between in/outgoing decisions
    conflict_count: usize,
}

/// Materializes one edge a.k.a arc from the decision diagram. It logically
/// connects two nodes and annotates the link with a decision and a cost.
/// Also, this formulation places the state on an edge in place of a node
#[derive(Debug, Clone, PartialEq, Eq)]
struct Edge<T> {
    /// The identifier of the node at the ∗∗source∗∗ of this edge.
    from: NodeId,
    /// The identifier of the node at the ∗∗destination∗∗ of this edge.
    to: NodeId,
    /// This is the decision label associated to this edge. It gives the
    /// information "what variable" is assigned to "what value".
    decision: Decision,
    /// This is the transition cost of making this decision from the state
    /// associated with the source node of this edge.
    cost: isize,
    /// The state associated to this edge
    state: Arc<T>,
}

/// The decision diagram in itself. This structure essentially keeps track
/// of the nodes composing the diagram as well as the edges connecting these
/// nodes in two vectors (enabling preallocation and good cache locality).
/// In addition to that, it also keeps track of the path (root_pa) from the
/// problem root to the root of this decision diagram (explores a sub problem).
/// The prev_l comprises information about the nodes that are currently being
/// expanded, next_l stores the information about the nodes from the next layer
/// and cut-set stores an exact cut-set of the DD.
/// Depending on the type of DD compiled, different cutset types will be used:
/// - Exact: no cut-set is needed since the DD is exact
/// - Restricted: the last exact layer is used as cut-set
/// - Relaxed: either the last exact layer of the frontier cut-set can be chosen
///            within the CompilationInput
#[derive(Debug, Clone)]
pub struct VectorMdd<T, const CUTSET_TYPE: CutsetType>
where
    T: Eq + PartialEq + Hash + Clone,
{
    /// All the nodes composing this decision diagram. The vector comprises
    /// nodes from all layers in the DD. All nodes
    /// belonging to one same layer form a vector.
    nodes: Vec<Vec<Node<T>>>,
    /// This vector stores the information about all edges connecting the nodes
    /// of the decision diagram.
    edges: Vec<Edge<T>>,
    /// Contains the nodes of the layer which is currently being expanded.
    /// This collection is only used during the unrolling of transition relation,
    /// and when merging nodes of a relaxed DD.
    prev_l: Vec<NodeId>,
    /// The nodes from the next layer; those are the result of an application
    /// of the transition function to a node in ‘prev_l‘.
    /// Note: next_l in itself is indexed on the state associated with nodes.
    /// The rationale being that two transitions to the same state in the same
    /// layer should lead to the same node. This indexation helps ensuring
    /// the uniqueness constraint in amortized O(1).
    next_l: FxHashMap<Arc<T>, NodeId>,
    /// The nodes from the final layer of the mdd.
    /// These are the nodes form which we extract solutions and bounds of the whole mdd.
    /// In top down constructions, these are simply the nodes in next_l after compilation
    /// But in refinement, we direcly fetch the last layer of the node as the hashmap implementation
    /// of next_l strongly assumes that any 2 transitions to the same state should lead to the same nodes
    /// which is not necessarily the case in the refinement procedure
    final_l: Vec<NodeId>,
    /// The depth of the layer currently being expanded
    curr_depth: usize,

    /// Keeps track of the decisions that have been taken to reach the root
    /// of this DD, starting from the problem root.
    path_to_root: Vec<Decision>,
    /// The identifier of the last exact layer (should this dd be inexact)
    lel: LayerId,
    /// The cut-set of the decision diagram (only maintained for relaxed dd)
    cutset: Vec<NodeId>,
    /// The identifier of the best terminal node of the diagram (None when the
    /// problem compiled into this dd is infeasible)
    best_node: Option<NodeId>,
    /// The identifier of the best exact terminal node of the diagram (None when
    /// no terminal node is exact)
    best_exact_node: Option<NodeId>,
    /// A flag set to true when no layer of the decision diagram has been
    /// restricted or relaxed
    is_exact: bool,
    /// A flag set to true when the longest r-t path of this decision diagram
    /// traverses no merged node (Exact Best Path Optimization aka EBPO).
    has_exact_best_path: bool,
}

// Tech note: WHY AM I USING MACROS HERE ?
// ---> Simply to avoid the need to fight the borrow checker

/// These macro retrieve an element of the dd by its id
macro_rules! get {
    (    node     $id:expr, $dd:expr) => {
        &$dd.nodes[$id.0][$id.1]
    };
    (mut node     $id:expr, $dd:expr) => {
        &mut $dd.nodes[$id.0][$id.1]
    };
    (    edge     $id:expr, $dd:expr) => {
        &$dd.edges[$id.0]
    };
    (mut edge     $id:expr, $dd:expr) => {
        &mut $dd.edges[$id.0]
    };
    (    edgelist $id:expr, $dd:expr) => {
        &$dd.edgelists[$id.0]
    };
    (mut edgelist $id:expr, $dd:expr) => {
        &mut $dd.edgelists[$id.0]
    };
    (    layer    $id:expr, $dd:expr) => {
        &$dd.nodes[$id.0]
    };
    (mut layer    $id:expr, $dd:expr) => {
        &mut $dd.nodes[$id.0]
    };
}

/// This macro performs an action for each incoming edge of a given node in the dd
macro_rules! foreachincoming {
    (edge of $id:expr, $dd:expr, $action:expr) => {
        let mut index = 0;
        while index < get!(node $id, $dd).incoming.len() {
            let edge_id = get!(node $id, $dd).incoming[index];
            let edge = get!(edge edge_id, $dd).clone();
            $action(edge);
            index += 1;
        }
    };
}

/// This macro performs an action for each outgoing edge of a given node in the dd
macro_rules! foreachoutgoing {
    (edge of $id:expr, $dd:expr, $action:expr) => {
        let mut index = 0;
        while index < get!(node $id, $dd).outgoing.len() {
            let edge_id = get!(node $id, $dd).outgoing[index];
            let edge = get!(edge edge_id, $dd).clone();
            $action(edge);
            index += 1;
        }
    };
}

/// This macro appends an edge to the list of edges adjacent to a given node
macro_rules! append_edge_to {
    ($dd:expr, $edge:expr) => {
        let new_eid = EdgeId($dd.edges.len());
        $dd.edges.push($edge);

        let parent = get!(mut node $edge.from, $dd);
        let parent_exact = parent.flags.is_exact();
        let value = parent.value_top.saturating_add($edge.cost);
        parent.outgoing.push(new_eid);

        let node = get!(mut node $edge.to, $dd);
        let exact = parent_exact & node.flags.is_exact();
        node.flags.set_exact(exact);
        node.incoming.push(new_eid);

        if value >= node.value_top {
            node.best = Some(new_eid);
            node.value_top = value;
        }
    };
}

/// This macro redirects an existing edge
macro_rules! redirect_edge {
    ($dd:expr, $e_id:expr, $edge:expr) => {

        let parent = get!(mut node $edge.from, $dd);
        let parent_exact = parent.flags.is_exact();
        let value = parent.value_top.saturating_add($edge.cost);
        parent.outgoing.push($e_id); // who are you?

        let node = get!(mut node $edge.to, $dd);
        let exact = parent_exact & node.flags.is_exact();
        node.flags.set_exact(exact);
        node.incoming.push($e_id);


        if value >= node.value_top {
            node.best = Some($e_id);
            node.value_top = value;
        }
    };
}

impl<T, const CUTSET_TYPE: CutsetType> Default for VectorMdd<T, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CUTSET_TYPE: CutsetType> DecisionDiagram for VectorMdd<T, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone,
{
    type State = T;

    fn compile(&mut self, input: &CompilationInput<Self::State>) -> Result<Completion, Reason> {
        self._compile(input)
    }

    fn refine(&mut self, input: &CompilationInput<Self::State>) -> Result<Completion, Reason> {
        self._refine(input)
    }

    fn is_exact(&self) -> bool {
        self.is_exact || self.has_exact_best_path
    }

    fn best_value(&self) -> Option<isize> {
        self._best_value()
    }

    fn best_solution(&self) -> Option<Solution> {
        self._best_solution()
    }

    fn best_exact_value(&self) -> Option<isize> {
        self._best_exact_value()
    }

    fn best_exact_solution(&self) -> Option<Solution> {
        self._best_exact_solution()
    }

    fn drain_cutset<F>(&mut self, func: F)
    where
        F: FnMut(SubProblem<Self::State>),
    {
        self._drain_cutset(func)
    }
}

impl<T, const CUTSET_TYPE: CutsetType> VectorMdd<T, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            //
            prev_l: vec![],
            next_l: Default::default(),
            final_l: vec![],
            curr_depth: 0,
            //
            path_to_root: vec![],
            lel: LayerId(0),
            cutset: vec![],
            best_node: None,
            best_exact_node: None,
            is_exact: false,
            has_exact_best_path: false,
        }
    }

    /// refinement starts from an existing diagram so unlike _clear we don't clear all properties
    /// onl those that will be reset by refinement
    fn _clear_for_refine(&mut self) {
        self.cutset.clear();
        self.path_to_root.clear();
        self.prev_l.clear();
        self.next_l.clear();
        self.final_l.clear();
        // do not reset lel, only start expanding from last point of lel to reduce redundant work
        self.best_node = None;
        self.best_exact_node = None;
        self.is_exact = false;
        self.has_exact_best_path = false;
    }

    fn _clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.prev_l.clear();
        self.next_l.clear();
        self.final_l.clear();
        self.path_to_root.clear();
        self.cutset.clear();
        self.lel = LayerId(0);
        self.best_node = None;
        self.best_exact_node = None;
        self.is_exact = false;
        self.has_exact_best_path = false;
    }

    fn _best_value(&self) -> Option<isize> {
        self.best_node.map(|id| get!(node id, self).value_top)
    }

    fn _best_solution(&self) -> Option<Vec<Decision>> {
        self.best_node.map(|id| self._best_path(id))
    }

    fn _best_exact_value(&self) -> Option<isize> {
        self.best_exact_node.map(|id| get!(node id, self).value_top)
    }

    fn _best_exact_solution(&self) -> Option<Vec<Decision>> {
        self.best_exact_node.map(|id| self._best_path(id))
    }

    fn _best_path(&self, id: NodeId) -> Vec<Decision> {
        Self::_best_path_partial_borrow(id, &self.path_to_root, &self.nodes, &self.edges)
    }

    fn _best_path_partial_borrow(
        id: NodeId,
        root_pa: &[Decision],
        nodes: &[Vec<Node<T>>],
        edges: &[Edge<T>],
    ) -> Vec<Decision> {
        let mut sol = root_pa.to_owned();
        let mut edge_id = nodes[id.0][id.1].best;
        while let Some(eid) = edge_id {
            let edge = edges[eid.0].clone();
            sol.push(edge.decision);
            edge_id = nodes[edge.from.0][edge.from.1].best;
        }
        sol
    }

    fn _compile(&mut self, input: &CompilationInput<T>) -> Result<Completion, Reason> {
        self._clear();
        self._initialize(input);

        let mut curr_l = vec![];
        while let Some(var) = input
            .problem
            .next_variable(self.curr_depth, &mut self.next_l.keys().map(|s| s.as_ref()))
        {
            // Did the cutoff kick in ?
            if input.cutoff.must_stop() {
                return Err(Reason::CutoffOccurred);
            }

            if !self._move_to_next_layer(input, &mut curr_l) {
                break;
            }
            for node_id in curr_l.iter() {
                let state = Arc::clone(&get!(node node_id, self).state);
                // self.nodes[node_id.0][node_id.1].state.clone();
                let rub = input.relaxation.fast_upper_bound(state.as_ref());
                get!(mut node node_id, self).rub = rub;
                let ub = rub.saturating_add(get!(node node_id, self).value_top);
                let is_exact_node = get!(node node_id, self).flags.is_exact();

                if ub > input.best_lb {
                    let is_must_keep_match = |decision| {
                        // TO DO: only call model if node exact and we are restricting
                        match input.comp_type {
                            CompilationType::Restricted => {
                                if is_exact_node {
                                    let must_keep_decision: Option<Decision> =
                                        input.problem.perform_ml_decision_inference(var, &state);
                                    match must_keep_decision {
                                        Some(x) => return x == decision,
                                        None => return false,
                                    };
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        }
                    };
                    input
                        .problem
                        .for_each_in_domain(var, state.as_ref(), &mut |decision| {
                            self._branch_on(
                                *node_id,
                                decision,
                                input.problem,
                                is_must_keep_match(decision),
                            )
                        })
                }
            }
            self.curr_depth += 1;
        }

        self._finalize(input);

        // /* 
        // ***************** visualise *****************
        // *********************************************
        let mut config = VizConfigBuilder::default().build().unwrap();
        // config.show_deleted = true;
        // config.show_deleted = true;
        config.group_merged = true;
        print!("after compile \n");
        let s = self.as_graphviz(&config);
        fs::write("incremental.dot", s).expect("Unable to write file");
        // ***********************************************
        // */

        Ok(Completion { 
            is_exact: self.is_exact(), 
            best_value: self.best_node.map(|n| get!(node n, self).value_top) 
        })
    }

    fn _refine(&mut self, input: &CompilationInput<T>) -> Result<Completion, Reason> {
        // clear parts of diagram reset by refinement
        self._clear_for_refine();

        self.path_to_root.extend_from_slice(&input.residual.path);
        self.curr_depth = input.residual.depth;

        // go layer by layer - we only start from the last exactlayer as anything before that is already completely refined
        // TODO: Sometimes we can also do bottom traversal - implement logic
        let mut curr_layer_id = self.lel.0;

        while curr_layer_id < self.nodes.len() {
            if input.cutoff.must_stop() {
                return Err(Reason::CutoffOccurred);
            }

            if !self._refine_curr_layer(input, curr_layer_id) {
                break;
            }

            curr_layer_id += 1;
            self.curr_depth += 1;
        }

        self._finalize(input);

        // /* 
        // ***************** visualise *****************
        // *********************************************
        let mut config = VizConfigBuilder::default().build().unwrap();
        // config.show_deleted = true;
        // config.show_deleted = true;
        config.group_merged = true;
        print!("after refine \n");
        let s = self.as_graphviz(&config);
        fs::write("incremental.dot", s).expect("Unable to write file");
        // ***********************************************
        // */

        Ok(Completion {
            is_exact: self.is_exact(),
            best_value: self.best_node.map(|n| get!(node n, self).value_top),
        })
    }

    fn _initialize(&mut self, input: &CompilationInput<T>) {
        self.path_to_root.extend_from_slice(&input.residual.path);

        let root_node_id = NodeId(0, 0);

        let root_node = Node {
            state: input.residual.state.clone(),
            value_top: input.residual.value,
            value_bot: isize::MIN,
            best: None,
            incoming: Vec::with_capacity(input.problem.nb_variables()), //TODO make this the number values variable can take
            outgoing: Vec::with_capacity(input.problem.nb_variables()),
            rub: isize::MAX,
            theta: None,
            flags: NodeFlags::new_exact(),
            depth: input.residual.depth,
            conflict_count: 0,
        };

        self.nodes.push(vec![root_node]);
        self.next_l
            .insert(input.residual.state.clone(), root_node_id);
        self.curr_depth = input.residual.depth;
    }

    fn _finalize(&mut self, input: &CompilationInput<T>) {
        self._extract_final_layer(input);
        self._find_best_node();
        self._finalize_exact(input);
        self._finalize_cutset(input);
        self._compute_local_bounds(input);
        self._compute_thresholds(input);
    }

    fn _drain_cutset<F>(&mut self, mut func: F)
    where
        F: FnMut(SubProblem<T>),
    {
        if let Some(best_value) = self.best_value() {
            for id in self.cutset.drain(..) {
                let node = get!(node id, self);

                if node.flags.is_marked() {
                    let rub = node.value_top.saturating_add(node.rub);
                    let locb = node.value_top.saturating_add(node.value_bot);
                    let ub = rub.min(locb).min(best_value);

                    func(SubProblem {
                        state: node.state.clone(),
                        value: node.value_top,
                        path: Self::_best_path_partial_borrow(
                            id,
                            &self.path_to_root,
                            &self.nodes,
                            &self.edges,
                        ),
                        ub,
                        depth: node.depth,
                    })
                }
            }
        }
    }

    #[allow(clippy::redundant_closure_call)]
    fn _compute_local_bounds(&mut self, input: &CompilationInput<T>) {
        if !self.is_exact && input.comp_type == CompilationType::Relaxed {
            // initialize last layer
            let last_layer_id = LayerId(self.nodes.len() - 1);
            let layer = get!(mut layer last_layer_id, self);
            for node in layer {
                node.value_bot = 0;
                node.flags.set_marked(true);
            }

            // traverse bottom-up
            // note: cache requires that all nodes have an associated locb. not only those below cutset
            for layer_id in (0..self.nodes.len()).rev() {
                let layer_len = self.nodes[layer_id].len();
                for id in 0..layer_len {
                    let id = NodeId(layer_id, id);
                    let node = get!(node id, self);
                    let value = node.value_bot;
                    if node.flags.is_marked() {
                        foreachincoming!(edge of id, self, |edge: Edge<T>| {
                            let using_edge = value.saturating_add(edge.cost);
                            let parent = get!(mut node edge.from, self);
                            parent.flags.set_marked(true);
                            parent.value_bot = parent.value_bot.max(using_edge);
                        });
                    }
                }
            }
        }
    }

    #[allow(clippy::redundant_closure_call)]
    fn _compute_thresholds(&mut self, input: &CompilationInput<T>) {
        if input.comp_type == CompilationType::Relaxed || self.is_exact {
            let mut best_known = input.best_lb;

            if let Some(best_exact_node) = self.best_exact_node {
                let best_exact_value = get!(mut node best_exact_node, self).value_top;
                best_known = best_known.max(best_exact_value);

                for id in &self.final_l {
                    if (CUTSET_TYPE == LAST_EXACT_LAYER && self.is_exact)
                        || (CUTSET_TYPE == FRONTIER && get!(node id, self).flags.is_exact())
                    {
                        get!(mut node id, self).theta = Some(best_known);
                    }
                }
            }

            for layer_id in (0..self.nodes.len()).rev() {
                for id in 0..self.nodes[layer_id].len() {
                    let id = NodeId(layer_id, id);
                    let node = get!(mut node id, self);

                    if node.flags.is_deleted() {
                        continue;
                    }

                    // ATTENTION: YOU WANT TO PROPAGATE THETA EVEN IF THE NODE WAS PRUNED BY THE CACHE
                    if !node.flags.is_pruned_by_cache() {
                        let tot_rub = node.value_top.saturating_add(node.rub);
                        if tot_rub <= best_known {
                            node.theta = Some(best_known.saturating_sub(node.rub));
                        } else if node.flags.is_cutset() {
                            let tot_locb = node.value_top.saturating_add(node.value_bot);
                            if tot_locb <= best_known {
                                let theta = node.theta.unwrap_or(isize::MAX);
                                node.theta =
                                    Some(theta.min(best_known.saturating_sub(node.value_bot)));
                            } else {
                                node.theta = Some(node.value_top);
                            }
                        } else if node.flags.is_exact() && node.theta.is_none() {
                            // large theta for dangling nodes
                            node.theta = Some(isize::MAX);
                        }

                        Self::_maybe_update_cache(node, input);
                    }
                    // only propagate if you have an actual threshold
                    if let Some(my_theta) = node.theta {
                        foreachincoming!(edge of id, self, |edge: Edge<T>| {
                            let parent = get!(mut node edge.from, self);
                            let theta  = parent.theta.unwrap_or(isize::MAX);
                            parent.theta = Some(theta.min(my_theta.saturating_sub(edge.cost)));
                        });
                    }
                }
            }
        }
    }

    fn _maybe_update_cache(node: &Node<T>, input: &CompilationInput<T>) {
        // A node can only be added to the cache if it belongs to the cutset or is above it
        if let Some(theta) = node.theta {
            if node.flags.is_above_cutset() {
                input.cache.update_threshold(
                    node.state.clone(),
                    node.depth,
                    theta,
                    !node.flags.is_cutset(),
                ) // if it is in the cutset it has not been explored !
            }
        }
    }

    fn _finalize_cutset(&mut self, input: &CompilationInput<T>) {
        // if self.lel.is_none() {
        //     self.lel = Some(LayerId(self.layers.len())); // all nodes of the DD are above cutset
        // }
        if input.comp_type == CompilationType::Relaxed || self.is_exact {
            match CUTSET_TYPE {
                LAST_EXACT_LAYER => {
                    self._compute_last_exact_layer_cutset(self.lel);
                }
                FRONTIER => {
                    self._compute_frontier_cutset();
                }
                _ => {
                    panic!("Only LAST_EXACT_LAYER and FRONTIER are supported so far")
                }
            }
        }
    }

    fn _compute_last_exact_layer_cutset(&mut self, lel: LayerId) {
        if !self.is_exact {
            let layer = get!(mut layer lel, self);
            for (id, node) in layer.iter_mut().enumerate() {
                self.cutset.push(NodeId(lel.0, id));
                node.flags
                    .add(NodeFlags::F_CUTSET | NodeFlags::F_ABOVE_CUTSET);
            }
        }

        // traverse bottom up to set the above cutset for all nodes in layers above LEL
        for layer_id in (0..self.nodes.len()).rev() {
            for id in 0..self.nodes[layer_id].len() {
                let id = NodeId(layer_id, id);
                let node = get!(mut node id, self);
                node.flags.set_above_cutset(true);
            }
        }
    }

    #[allow(clippy::redundant_closure_call)]
    fn _compute_frontier_cutset(&mut self) {
        // traverse bottom-up
        for layer_id in (0..self.nodes.len()).rev() {
            for id in 0..self.nodes[layer_id].len() {
                let id = NodeId(layer_id, id);
                let node = get!(mut node id, self);

                if node.flags.is_exact() {
                    node.flags.set_above_cutset(true);
                } else {
                    foreachincoming!(edge of id, self, |edge: Edge<T>| {
                        let parent = get!(mut node edge.from, self);
                        if parent.flags.is_exact() && !parent.flags.is_cutset() {
                            self.cutset.push(edge.from);
                            parent.flags.set_cutset(true);
                        }
                    });
                }
            }
        }
    }

    fn _extract_final_layer(&mut self, input: &CompilationInput<T>) {
        self.final_l.clear();
        match input.comp_strategy {
            CompilationStrategy::TopDown => {
                for (_, id) in self.next_l.drain() {
                    self.final_l.push(id);
                }
            }
            CompilationStrategy::Refinement => {
                let final_layer_id = LayerId(self.nodes.len() - 1);
                // self.final_l = (0..get!(layer final_layer_id, self).len()).map(|id| NodeId(final_layer_id.0, id)).collect::<Vec<_>>();
                self.final_l = (0..get!(layer final_layer_id, self).len())
                    .filter_map(|id| {
                        (!get!(node NodeId(final_layer_id.0, id), self)
                            .flags
                            .is_deleted()
                            && !get!(node NodeId(final_layer_id.0, id), self)
                                .flags
                                .is_pruned_by_dominance()
                            && !get!(node NodeId(final_layer_id.0, id), self)
                                .flags
                                .is_pruned_by_cache())
                        .then_some(NodeId(final_layer_id.0, id))
                    })
                    .collect::<Vec<_>>();
            }
        }
    }

    fn _find_best_node(&mut self) {
        self.best_node = self
            .final_l
            .iter()
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
        self.best_exact_node = self
            .final_l
            .iter()
            .filter(|id| get!(node id, self).flags.is_exact())
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
    }

    fn _finalize_exact(&mut self, input: &CompilationInput<T>) {
        self.is_exact = self.lel.0 == self.nodes.len() - 1;
        self.has_exact_best_path = matches!(input.comp_type, CompilationType::Relaxed)
            && self._has_exact_best_path(self.best_node);

        if self.has_exact_best_path {
            self.best_exact_node = self.best_node;
        }
    }

    fn _has_exact_best_path(&self, node: Option<NodeId>) -> bool {
        if let Some(node_id) = node {
            let n = get!(node node_id, self);
            if n.flags.is_exact() {
                true
            } else {
                !n.flags.is_relaxed()
                    && self._has_exact_best_path(n.best.map(|e| get!(edge e, self).from))
            }
        } else {
            true
        }
    }

    fn _move_to_next_layer(
        &mut self,
        input: &CompilationInput<T>,
        curr_l: &mut Vec<NodeId>,
    ) -> bool {
        self.prev_l.clear();

        for id in curr_l.drain(..) {
            self.prev_l.push(id);
        }
        for (_, id) in self.next_l.drain() {
            curr_l.push(id);
        }

        if curr_l.is_empty() {
            false
        } else {
            if self.nodes.len() > 1 {
                self._filter_with_cache(input, curr_l);
            }
            self._filter_with_dominance(input, curr_l);

            self._squash_if_needed(input, curr_l);

            // create a new layer in nodes vector for next expansion
            self.nodes.push(vec![]);

            true
        }
    }

    fn _refine_curr_layer(&mut self, input: &CompilationInput<T>, curr_layer_id: usize) -> bool {
        // fetch layer nodes
        // only collect those that are not deleted
        let mut curr_l = get!(layer LayerId(curr_layer_id), self)
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                (!node.flags.is_deleted()).then_some(NodeId(curr_layer_id, id))
            })
            .collect::<Vec<_>>();

        if curr_l.is_empty() {
            // TODO I return false as soon as a layer is empty? The refinement process stops beacuse why? Does that mean infeasible then?
            false
        } else {
            if curr_layer_id != 0 {
                self._filter_with_cache(input, &mut curr_l);
                 // /* 
                // ***************** visualise *****************
                // *********************************************
                let mut config = VizConfigBuilder::default().build().unwrap();
                // config.show_deleted = true;
                // config.show_deleted = true;
                config.group_merged = true;
                print!("before split layer {curr_layer_id}\n");
                let s = self.as_graphviz(&config);
                fs::write("incremental.dot", s).expect("Unable to write file"); 
                // *************************************************************
                // */
                self._filter_constraints(input, &mut curr_l); // the root has no incoming so cannot filter
                 // /* 
                // ***************** visualise *****************
                // *********************************************
                let mut config = VizConfigBuilder::default().build().unwrap();
                // config.show_deleted = true;
                // config.show_deleted = true;
                config.group_merged = true;
                print!("before split layer {curr_layer_id}\n");
                let s = self.as_graphviz(&config);
                fs::write("incremental.dot", s).expect("Unable to write file"); 
                // *************************************************************
                // */
            }
            // filter the incoming edge states instead
            self._filter_with_dominance_edge_based(input, &mut curr_l);
             // /* 
            // ***************** visualise *****************
            // *********************************************
            let mut config = VizConfigBuilder::default().build().unwrap();
            // config.show_deleted = true;
            // config.show_deleted = true;
            config.group_merged = true;
            print!("before split layer {curr_layer_id}\n");
            let s = self.as_graphviz(&config);
            fs::write("incremental.dot", s).expect("Unable to write file"); 
            // *************************************************************
            // */

            self._stretch_if_needed(input, &mut curr_l, curr_layer_id);

            // self._filter_with_dominance(input, &mut curr_l);

            // if curr_layer_id != 0 {
            //     self._filter_with_cache(input, &mut curr_l);
            //     self._filter_constraints(input, &mut curr_l); // the root has no incoming so cannot filter
            // }

            true
        }
    }

    fn _filter_with_dominance(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        curr_l.sort_unstable_by(|a, b| {
            input
                .dominance
                .cmp(
                    get!(node a, self).state.as_ref(),
                    get!(node a, self).value_top,
                    get!(node b, self).state.as_ref(),
                    get!(node b, self).value_top,
                )
                .reverse()
        });
        curr_l.retain(|id| {
            let node = get!(mut node id, self);
            if node.flags.is_exact() {
                let DominanceCheckResult {
                    dominated,
                    threshold,
                } = input.dominance.is_dominated_or_insert(
                    node.state.clone(),
                    node.depth,
                    node.value_top,
                );
                if dominated {
                    node.flags.set_pruned_by_dominance(true);
                    node.outgoing = vec![];
                    node.theta = threshold;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
    }

    fn _filter_with_dominance_edge_based(
        &mut self,
        input: &CompilationInput<T>,
        curr_l: &mut Vec<NodeId>,
    ) {
        let mut curr_in_edges = vec![];
        for node_id in curr_l.iter() {
            curr_in_edges.extend(
                get!(node node_id, self)
                    .incoming
                    .iter()
                    .filter_map(|x| {
                        (!get!(node get!(edge x, self).from, self).flags.is_deleted()
                            && !get!(node get!(edge x, self).from, self)
                                .flags
                                .is_pruned_by_dominance()
                            && !get!(node get!(edge x, self).from, self)
                                .flags
                                .is_pruned_by_cache()
                            && !input.problem.filter(
                                &get!(node get!(edge x, self).from, self).state,
                                &get!(edge x, self).decision,
                            ))
                        .then_some(*x)
                    })
                    .collect::<Vec<_>>(),
            );
        }

        curr_in_edges.sort_unstable_by(|a, b| {
            input
                .dominance
                .cmp(
                    get!(edge a, self).state.as_ref(),
                    get!(node get!(edge a, self).from,self).value_top + get!(edge a, self).cost,
                    get!(edge b, self).state.as_ref(),
                    get!(node get!(edge b, self).from,self).value_top + get!(edge b, self).cost,
                )
                .reverse()
        });
        curr_in_edges.retain(|id| {
            let src_id = get!(edge id, self).from;
            let dst_id = get!(edge id, self).to;
            let cost = get!(edge id, self).cost;

            let mut src_outgoing = get!(node src_id, self)
                .outgoing
                .iter()
                .map(|x| *x)
                .collect::<Vec<_>>();
            let mut dst_incoming = get!(node dst_id, self)
                .incoming
                .iter()
                .map(|x| *x)
                .collect::<Vec<_>>();

            if get!(node src_id,self).flags.is_exact() {
                let DominanceCheckResult {
                    dominated,
                    threshold,
                } = input.dominance.is_dominated_or_insert(
                    get!(edge id, self).state.clone(),
                    get!(node dst_id, self).depth,
                    get!(node src_id, self).value_top + cost,
                );
                if dominated {
                    //TODO, what happens here
                    //remove edge from incoming/outgoing
                    src_outgoing.retain(|e_id| e_id != id);
                    dst_incoming.retain(|e_id| e_id != id);
                    if src_outgoing.is_empty() {
                        // set node as deleted and remove from layer
                        get!(mut node src_id, self)
                            .flags
                            .set_pruned_by_dominance(true);
                    }
                    if dst_incoming.is_empty() {
                        // set node as deleted and remove from layer
                        get!(mut node dst_id, self)
                            .flags
                            .set_pruned_by_dominance(true);
                        //TODO, how does threshold work
                        // only set it for destination node
                        get!(mut node dst_id, self).theta = threshold;
                    }
                    get!(mut node src_id, self).outgoing = src_outgoing;
                    get!(mut node dst_id, self).incoming = dst_incoming;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });

        curr_l.retain(|&x| !get!(mut node x, self).flags.is_pruned_by_dominance());
    }

    fn _filter_with_cache(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        curr_l.retain(|id| {
            let node = get!(mut node id, self);
            let threshold = input.cache.get_threshold(node.state.as_ref(), node.depth);
            if let Some(threshold) = threshold {
                if node.value_top > threshold.value {
                    true
                } else {
                    node.flags.set_pruned_by_cache(true);
                    node.theta = Some(threshold.value); // set theta for later propagation
                    false
                }
            } else {
                true
            }
        });
    }

    //TODO we should actually compute this filter at the point of edge creation so we don't do this later. FIX!
    fn _filter_constraints(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        for node_id in curr_l.iter() {
            // This is to prevent fighting the borrow checker
            let inbound = get!(node node_id, self)
                .incoming
                .iter()
                .filter_map(|x| {
                    (!get!(node get!(edge x, self).from, self).flags.is_deleted()
                        && !get!(node get!(edge x, self).from, self)
                            .flags
                            .is_pruned_by_dominance()
                        && !get!(node get!(edge x, self).from, self)
                            .flags
                            .is_pruned_by_cache()
                        && !input.problem.filter(
                            &get!(node get!(edge x, self).from, self).state,
                            &get!(edge x, self).decision,
                        ))
                    .then_some(*x)
                })
                .collect::<Vec<_>>();
            let inbound_unfiltered = get!(node node_id, self).incoming.iter().collect::<Vec<_>>();
            // let outbound = get!(node node_id, self).outgoing.iter()
            //                                                 .filter_map(|x| {
            //                                                     (!get!(node node_id, self).flags.is_deleted() && !input.problem.filter(&get!(node node_id, self).state, &get!(edge x, self).decision)).then_some(*x)
            //                                                 })
            //                                                 .collect::<Vec<_>>();

            // Ideal goal syntax but borrow check problems
            // get!(mut node node_id, self).incoming.retain(|x| !get!(node get!(edge x, self).from, self).flags.is_deleted() && !input.problem.filter(&get!(node get!(edge x, self).from, self).state, &get!(edge x, self).decision));
            // get!(mut node node_id, self).outgoing.retain(|x| !get!(node node_id, self).flags.is_deleted() && !input.problem.filter(&get!(node node_id, self).state, &get!(edge x, self).decision));

            //TODO if outbound edges all detached, set node as deleted-- basically if outbound is nil? I think
            if inbound.is_empty() {
                // set node as deleted and remove from layer
                get!(mut node node_id, self).flags.set_deleted(true);
                get!(mut node node_id, self).outgoing = vec![];
            } else {
                get!(mut node node_id, self).incoming = inbound;
                // get!(mut node node_id, self).outgoing = outbound;
            }
        }
        curr_l.retain(|&x| !get!(mut node x, self).flags.is_deleted());
    }

    fn _branch_on(
        &mut self,
        from_id: NodeId,
        decision: Decision,
        problem: &dyn Problem<State = T>,
        must_keep: bool,
    ) {
        let state = get!(node from_id, self).state.as_ref();
        let next_state = Arc::new(problem.transition(state, decision));
        let cost = problem.transition_cost(state, next_state.as_ref(), decision);
        let next_layer_id = from_id.0 + 1;

        match self.next_l.entry(next_state.clone()) {
            Entry::Vacant(e) => {
                let parent = get!(node from_id, self).clone();
                let node_id = NodeId(next_layer_id, self.nodes[next_layer_id].len());
                let mut flags = NodeFlags::new_exact();
                flags.set_exact(parent.flags.is_exact());
                flags.set_must_keep(must_keep);

                self.nodes[next_layer_id].push(Node {
                    state: next_state.clone(),
                    value_top: parent.value_top.saturating_add(cost),
                    value_bot: isize::MIN,
                    //
                    best: None,
                    incoming: Vec::with_capacity(problem.nb_variables()), //TODO make this the number values variable can take
                    outgoing: Vec::with_capacity(problem.nb_variables()), //TODO make this the number values variable can take
                    //
                    rub: isize::MAX,
                    theta: None,
                    flags,
                    depth: parent.depth + 1,
                    conflict_count: 0,
                });
                append_edge_to!(
                    self,
                    Edge {
                        from: from_id,
                        to: node_id,
                        decision,
                        cost,
                        state: next_state.clone()
                    }
                );
                e.insert(node_id);
            }
            Entry::Occupied(e) => {
                let node_id = *e.get();
                append_edge_to!(
                    self,
                    Edge {
                        from: from_id,
                        to: node_id,
                        decision,
                        cost,
                        state: next_state.clone()
                    }
                );
            }
        }
    }

    fn _squash_if_needed(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        if self.nodes.len() > 1 {
            match input.comp_type {
                CompilationType::Exact => {
                    /* do nothing: you want to explore the complete DD */
                    self._maybe_save_lel(); /* every layer is exact */
                }
                CompilationType::Restricted => {
                    if curr_l.len() > input.max_width {
                        self._restrict(input, curr_l)
                    } else {
                        self._maybe_save_lel();
                    }
                }
                CompilationType::Relaxed => {
                    if curr_l.len() > input.max_width {
                        self._relax(input, curr_l)
                    } else {
                        self._maybe_save_lel();
                    }
                }
            }
        }
    }

    fn _stretch_if_needed(
        &mut self,
        input: &CompilationInput<T>,
        curr_l: &mut Vec<NodeId>,
        curr_layer_id: usize,
    ) {
        if curr_layer_id > 0 {
            /* we dont want to stretch the root of the subproblem */
            match input.comp_type {
                CompilationType::Restricted => { /* refinement does not build restrictions */ }
                _ => {
                    /* for both exact and realaxed we just keep splitting and filtering */
                    let mut fully_split = false;
                    while curr_l.len() < input.max_width {
                        if input.binary_split {
                            fully_split = self._split_binary(input, curr_l, curr_layer_id);
                        } else {
                            fully_split = self._split(input, curr_l, curr_layer_id);
                        }

                        // set rub
                        for node_id in curr_l.iter() {
                            let state = Arc::clone(&get!(node node_id, self).state);
                            let rub = input.relaxation.fast_upper_bound(state.as_ref());
                            get!(mut node node_id, self).rub = rub;
                        }

                        if fully_split {
                            break;
                        } // nothing to stretch
                    }
                    if fully_split
                        && self.lel.0 == curr_layer_id - 1
                        && curr_l
                            .iter()
                            .filter(|x| !get!(node x, self).flags.is_deleted())
                            .all(|x| get!(node x, self).flags.is_exact())
                    {
                        self.lel = LayerId(curr_layer_id); // lel was the previous layer
                    }
                }
            }
        }
    }

    fn _maybe_save_lel(&mut self) {
        let last_layer_id = self.nodes.len() - 1;
        if self.lel.0 == last_layer_id - 1 {
            self.lel = LayerId(last_layer_id); // lel was the previous layer
        }
    }

    fn _restrict(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        // curr_l.sort_unstable_by(|a, b| {
        //     get!(node a, self).value_top
        //         .cmp(&get!(node b, self).value_top)
        //         .then_with(|| input.ranking.compare(get!(node a, self).state.as_ref(), get!(node b, self).state.as_ref()))
        //         .reverse()
        // }); // reverse because greater means more likely to be kept

        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .flags
                .is_must_keep()
                .cmp(&get!(node b, self).flags.is_must_keep()) // make sure any must_keep nodes are higher in the ranking
                .then_with(|| {
                    get!(node a, self)
                        .value_top
                        .cmp(&get!(node b, self).value_top)
                        .then_with(|| {
                            input.ranking.compare(
                                get!(node a, self).state.as_ref(),
                                get!(node b, self).state.as_ref(),
                            )
                        })
                })
                .reverse()
        }); // reverse because greater means more likely to be kept

        for drop_id in curr_l.iter().skip(input.max_width).copied() {
            get!(mut node drop_id, self).flags.set_deleted(true);
        }

        curr_l.truncate(input.max_width);
    }

    #[allow(clippy::redundant_closure_call)]
    fn _relax(&mut self, input: &CompilationInput<T>, curr_l: &mut Vec<NodeId>) {
        // curr_l.sort_unstable_by(|a, b| {
        //     get!(node a, self).value_top
        //         .cmp(&get!(node b, self).value_top)
        //         .then_with(|| input.ranking.compare(get!(node a, self).state.as_ref(), get!(node b, self).state.as_ref()))
        //         .reverse()
        // }); // reverse because greater means more likely to be kept

        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .flags
                .is_must_keep()
                .cmp(&get!(node b, self).flags.is_must_keep()) // make sure any must_keep nodes are higher in the ranking
                .then_with(|| {
                    get!(node a, self)
                        .value_top
                        .cmp(&get!(node b, self).value_top)
                        .then_with(|| {
                            input.ranking.compare(
                                get!(node a, self).state.as_ref(),
                                get!(node b, self).state.as_ref(),
                            )
                        })
                })
                .reverse()
        }); // reverse because greater means more likely to be kept

        //--
        let (keep, merge) = curr_l.split_at_mut(input.max_width - 1);
        let merged = Arc::new(
            input
                .relaxation
                .merge(&mut merge.iter().map(|id| get!(node id, self).state.as_ref())),
        );

        let recycled = keep
            .iter()
            .find(|id| get!(node * id, self).state.eq(&merged))
            .copied();

        let merged_id = recycled.unwrap_or_else(|| {
            let last_layer_id = self.nodes.len() - 1;
            let node_id = NodeId(last_layer_id, self.nodes[last_layer_id].len());
            let depth = get!(node merge[0], self).depth;
            self.nodes[last_layer_id].push(Node {
                state: merged.clone(),
                value_top: isize::MIN,
                value_bot: isize::MIN,
                best: None,                                                 // yet
                incoming: Vec::with_capacity(input.problem.nb_variables()), //TODO make this the number values variable can take
                outgoing: Vec::with_capacity(input.problem.nb_variables()), //TODO make this the number values variable can take
                //
                rub: isize::MAX,
                theta: None,
                flags: NodeFlags::new_relaxed(),
                depth,
                conflict_count: 0,
            });
            node_id
        });

        get!(mut node merged_id, self).flags.set_relaxed(true);

        for drop_id in merge {
            get!(mut node drop_id, self).flags.set_deleted(true);

            foreachincoming!(edge of drop_id, self, |edge: Edge<T>| {
                // let edge = get!(edge edge_id, self);
                let src   = get!(node edge.from, self).state.as_ref();
                let dst   = get!(node edge.to,   self).state.as_ref();
                let rcost = input.relaxation.relax(src, dst, merged.as_ref(), edge.decision, edge.cost);

                append_edge_to!(self, Edge {
                    from: edge.from,
                    to: merged_id,
                    decision: edge.decision,
                    cost: rcost,
                    state: edge.state.clone() // it keeps the same state as it was, merging does not change edge state
                });
            });
        }

        if recycled.is_some() {
            curr_l.truncate(input.max_width);
            let saved_id = curr_l[input.max_width - 1];
            get!(mut node saved_id, self).flags.set_deleted(false);
        } else {
            curr_l.truncate(input.max_width - 1);
            curr_l.push(merged_id);
        }
    }

    fn _split(
        &mut self,
        input: &CompilationInput<T>,
        curr_l: &mut Vec<NodeId>,
        curr_layer_id: usize,
    ) -> bool {

        // /* 
        // ***************** visualise *****************
        // *********************************************
        let mut config = VizConfigBuilder::default().build().unwrap();
        // config.show_deleted = true;
        // config.show_deleted = true;
        config.group_merged = true;
        print!("before split layer {curr_layer_id}\n");
        let s = self.as_graphviz(&config);
        fs::write("incremental.dot", s).expect("Unable to write file"); 
        // *************************************************************
        // */

        // order vec node by ranking

        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| {
                    get!(node a, self)
                        .conflict_count
                        .cmp(&get!(node b, self).conflict_count)
                })
        }); // no reverse because greater means more likely to be split

        // send all inbound to be split into n nodes
        let mut how_many = input.max_width;

        let mut to_split = curr_l.clone();
        // Don't split already exact nodes
        to_split.retain(|node_id: &NodeId| {
            if get!(node node_id, self).flags.is_exact() {
                how_many -= 1;
                false
            } else {
                true
            }
        });

        // Done with figuring out which/how many to split
        let how_many = how_many;
        let to_split = to_split;

        let mut inbound_edges: Vec<(usize, isize, &Decision, &T)> = vec![];
        // Get all inbound edges -- already filtered at this point
        for node_id in &to_split {
            inbound_edges.extend(
                get!(node node_id, self)
                    .incoming
                    .iter()
                    .map(|x| {
                        (
                            x.0,
                            get!(node self.edges[x.0].from,self).value_top + self.edges[x.0].cost,
                            &self.edges[x.0].decision,
                            self.edges[x.0].state.as_ref(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            //Delete split node
            get!(mut node node_id, self).flags.set_deleted(true);
        }

        let fully_split = how_many >= inbound_edges.len();
        let split_state_edges = if fully_split {
            inbound_edges
                .into_iter()
                .map(|(id, _, _, _)| Vec::from([id]))
                .collect()
        } else {
            input
                .problem
                .split_edges(&mut inbound_edges.into_iter(), how_many)
        };

        //TODO: here send things resulting in same state to same node
        let split_states: Vec<(Arc<T>, Vec<usize>)> = split_state_edges
            .into_iter()
            .map(|cluster| {
                let merged = self._merge_states_from_incoming_edges(input, &cluster);
                (merged, cluster)
            })
            .collect();

        //for each split state, create new nodes and redirect outbound edges
        let mut new_nodes =
            self._redirect_edges_after_split(input, &split_states, LayerId(curr_layer_id));
        // println!("redirected edges layer {:?} \n", curr_layer_id);
        curr_l.retain(|&x| !to_split.contains(&x));
        curr_l.append(&mut new_nodes);

        for node_id in &to_split {
            //Delete split node
            get!(mut node node_id, self).flags.set_deleted(true);
        }

        // /* 
        // ***************** visualise *****************
        // *********************************************
        let mut config = VizConfigBuilder::default().build().unwrap();
        // config.show_deleted = true;
        // config.show_deleted = true;
        config.group_merged = true;
        print!("after split layer {curr_layer_id}\n");
        let s = self.as_graphviz(&config);
        fs::write("incremental.dot", s).expect("Unable to write file");
        // *************************************************
        // */

        fully_split
    }

    fn _split_binary(
        &mut self,
        input: &CompilationInput<T>,
        curr_l: &mut Vec<NodeId>,
        curr_layer_id: usize,
    ) -> bool {
        // order vec node by ranking
        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| {
                    get!(node a, self)
                        .conflict_count
                        .cmp(&get!(node b, self).conflict_count)
                })
        }); // no reverse because greater means more likely to be split

        // select worst node and split
        let mut index = curr_l.len();
        while index > 0 {
            let node_to_split_id = curr_l[index - 1];
            let node_to_split = get!(node node_to_split_id, self);

            // collect inbound and outbound edges from linked list structure
            let mut inbound_edges = node_to_split
                .incoming
                .iter()
                .map(|x| {
                    (
                        x.0,
                        get!(node self.edges[x.0].from,self).value_top + self.edges[x.0].cost,
                        &self.edges[x.0].decision,
                        self.edges[x.0].state.as_ref(),
                    )
                })
                .collect::<Vec<_>>();

            if inbound_edges.len() > 1 {
                let split_states = self._split_node(input, &mut inbound_edges.into_iter());

                //Delete split node
                get!(mut node node_to_split_id, self)
                    .flags
                    .set_deleted(true);

                let mut new_nodes =
                    self._redirect_edges_after_split(input, &split_states, LayerId(curr_layer_id));

                curr_l.remove(index - 1);
                curr_l.append(&mut new_nodes);

                // let mut config = VizConfigBuilder::default().build().unwrap();
                // // config.show_deleted = true;
                // // config.show_deleted = true;
                // config.group_merged = true;
                // print!("after split layer {curr_layer_id}\n");
                // let s = self.as_graphviz(&config);
                // fs::write("incremental.dot", s).expect("Unable to write file");

                return false;
            } else {
                index -= 1;
                if inbound_edges.is_empty() {
                    get!(mut node node_to_split_id, self)
                        .flags
                        .set_deleted(true);
                }
            }
        }
        // let mut config = VizConfigBuilder::default().build().unwrap();
        // // config.show_deleted = true;
        // config.group_merged = true;
        // print!("after split layer {curr_layer_id}\n");
        // let s = self.as_graphviz(&config);
        // fs::write("incremental.dot", s).expect("Unable to write file");

        true
    }

    fn _split_node(
        &self,
        input: &CompilationInput<T>,
        inbound_edges: &mut dyn Iterator<Item = (usize, isize, &Decision, &T)>,
    ) -> Vec<(Arc<T>, Vec<usize>)> {
        // by default tries to split into 2
        let split_state_edges = input.problem.split_edges(inbound_edges, 2);

        split_state_edges
            .into_iter()
            .map(|cluster| {
                let merged = self._merge_states_from_incoming_edges(input, &cluster);
                (merged, cluster)
            })
            .collect()
    }

    fn _merge_states_from_incoming_edges(
        &self,
        input: &CompilationInput<T>,
        inbound_edges: &Vec<usize>,
    ) -> Arc<T> {
        // merge them as they go to the same state actually
        let merged = Arc::new(
            input.relaxation.merge(
                &mut inbound_edges
                    .iter()
                    .map(|x| get!(edge EdgeId(*x), self).state.as_ref()),
            ),
        );
        merged
    }

    fn _redirect_edges_after_split(
        &mut self,
        input: &CompilationInput<T>,
        split_states: &Vec<(Arc<T>, Vec<usize>)>,
        curr_layer_id: LayerId,
    ) -> Vec<NodeId> {
        let mut new_nodes = Vec::with_capacity(split_states.len());
        let mut outgoing_nodes_to_update = FxHashSet::default();

        for (state, incoming_edges) in split_states {
            // create new node
            // TODO: what is the depth now? This is a round about way to get depth, Also (rightly) assumes all nodes at incoming edges are at same depth
            let depth = get!(node(get!(edge EdgeId(incoming_edges[0]),self).to), self).depth;
            let curr_layer = get!(mut layer curr_layer_id, self);
            curr_layer.push(Node {
                state: Arc::clone(state),
                value_top: isize::MIN,
                value_bot: isize::MIN,
                best: None,                                                 // yet
                incoming: Vec::with_capacity(input.problem.nb_variables()), //TODO make this the number values variable can take
                outgoing: Vec::with_capacity(input.problem.nb_variables()), //TODO make this the number values variable can take
                //
                rub: isize::MAX,
                theta: None,
                // flags: NodeFlags::new_relaxed(),
                flags: if incoming_edges.len() == 1 {
                    NodeFlags::new_exact()
                } else {
                    NodeFlags::new_relaxed()
                },
                depth,
                conflict_count: 0,
            });

            let split_id = NodeId(curr_layer_id.0, curr_layer.len() - 1);
            new_nodes.push(split_id);

            let mut outbound_edges = FxHashSet::default();
            for edge_id in incoming_edges {
                let outbound_node = get!(edge EdgeId(*edge_id), self).to;
                outbound_edges.extend(get!(node outbound_node, self).outgoing.iter());
            }

            // TODO remove duplicated outbound edges via dedup https://doc.rust-lang.org/std/vec/struct.Vec.html#method.dedup
            self._redirect_incoming_edges(input, incoming_edges, &state, split_id, &outbound_edges);
            // redirect outgoing edges
            if !get!(node split_id, self).flags.is_deleted() {
                outgoing_nodes_to_update.extend(self._redirect_outgoing_edges(
                    input,
                    &outbound_edges,
                    &state,
                    split_id,
                ));
            }
        }
        //TODO: can this be more efficient?
        // update outgoing node
        // update outgoing node edges
        // for outgoing_node_id in outgoing_nodes_to_update {
        //     self.update_node(input, outgoing_node_id);
        // }

        new_nodes
    }

    fn _redirect_incoming_edges(
        &mut self,
        input: &CompilationInput<T>,
        edges_to_append: &Vec<usize>,
        state: &&Arc<T>,
        split_id: NodeId,
        outbound_edges: &FxHashSet<EdgeId>,
    ) {
        let mut conflict_count = 0;
        for edge_id in edges_to_append {
            // update edge state
            let edge = get!(edge EdgeId(*edge_id), self).clone();
            let from_node = get!(node edge.from, self);
            let new_state = Arc::new(input.problem.transition(
                from_node.state.as_ref(),
                get!(edge EdgeId(*edge_id), self).decision,
            ));

            // Here cost changes because the node has a different origin now
            let cost = input.problem.transition_cost(
                from_node.state.as_ref(),
                state.as_ref(),
                get!(edge EdgeId(*edge_id), self).decision,
            );
            // update edge
            get!(mut edge EdgeId(*edge_id), self).to = split_id;
            get!(mut edge EdgeId(*edge_id), self).cost = cost;
            get!(mut edge EdgeId(*edge_id), self).state = new_state;

            redirect_edge!(self, EdgeId(*edge_id), get!(edge EdgeId(*edge_id), self));

            for outbound_edge in outbound_edges {
                if input.problem.check_conflict(
                    &edge.decision,
                    &get!(edge EdgeId(outbound_edge.0), self).decision,
                ) {
                    conflict_count += 1;
                }
            }
        }
        // set node to exact if some and all and relaxed otherwise
        // exactness is further corrected by edge additon which checks parent exactness too
        // TODO: when is a node exact now?

        let split_node = get!(mut node split_id, self);
        split_node.conflict_count = conflict_count;
    }
    //TODO these are not checked for duplications? What if there's already an edge like this to that destination node?
    fn _redirect_outgoing_edges(
        &mut self,
        input: &CompilationInput<T>,
        outbound_edges: &FxHashSet<EdgeId>,
        state: &&Arc<T>,
        split_id: NodeId,
    ) -> Vec<NodeId> {
        //TODO replicate all edges outbound - we need to recalculate the decision states because it changes upon split - also filter infeasible outbounds
        let mut outgoing_nodes_to_update = Vec::with_capacity(outbound_edges.len());
        for edge_id in outbound_edges {
            let e = get!(edge edge_id,self);
            //TODO filter constraints instead - indeed filter here
            if !input.problem.filter(&state, &e.decision) {
                outgoing_nodes_to_update.push(e.to);

                // update edge state
                let new_state = Arc::new(input.problem.transition(state, e.decision));
                
                // Here cost changes because the node has a different origin now
                let cost = input.problem.transition_cost(
                    state.as_ref(),
                    new_state.as_ref(),
                    get!(edge edge_id, self).decision,
                );

                append_edge_to!(
                    self,
                    Edge {
                        from: split_id,
                        to: get!(edge edge_id, self).to,
                        decision: get!(edge edge_id, self).decision,
                        cost: cost,
                        state: new_state.clone()
                    }
                );
                // get!(mut edge edge_id, self).from = split_id;
                // get!(mut edge edge_id, self).cost = cost;
                // get!(mut edge edge_id, self).state = new_state;
                // redirect_edge!(self,*edge_id,get!(edge edge_id, self));

                //TODO update node properties that were formerly updated in append edge to
            }
        }
        outgoing_nodes_to_update
    }

    fn update_node(&mut self, input: &CompilationInput<T>, node_id: NodeId) {
        // let node_to_update = get!(node node_id,self);

        let inbound_edges = get!(node node_id, self)
            .incoming
            .iter()
            .map(|x| x.0)
            .collect();
        //update node attributes
        let merged = self._merge_states_from_incoming_edges(input, &inbound_edges);
        self.clear_node(node_id, merged);
        // TODO but also filter inbound edges before recalc
        for e_id in &inbound_edges {
            let in_edge = get!(edge EdgeId(*e_id),self);
            if !get!(node in_edge.from, self).flags.is_deleted() {
                let parent = get!(mut node in_edge.from, self);
                let parent_exact = parent.flags.is_exact();
                let value = parent.value_top.saturating_add(in_edge.cost);

                let node = get!(mut node in_edge.to, self);
                let exact = parent_exact & node.flags.is_exact();
                node.flags.set_exact(exact);

                if value >= node.value_top {
                    node.best = Some(EdgeId(*e_id));
                    node.value_top = value;
                }

                foreachoutgoing!(edge of node_id, self, |out_edge: Edge<T>| {
                    if input
                        .problem
                        .check_conflict(&in_edge.decision, &out_edge.decision)
                    {
                        get!(mut node node_id, self).conflict_count += 1;
                    }
                });
            }
        }
    }

    fn clear_node(&mut self, node_id: NodeId, state: Arc<T>) {
        let node = get!(mut node node_id, self);
        node.state = state;
        node.value_top = isize::MIN;
        node.value_bot = isize::MIN;
        node.theta = None;
        node.best = None;
        node.conflict_count = 0;
    }
}

// ############################################################################
// #### VISUALIZATION #########################################################
// ############################################################################
/// This is how you configure the output visualisation e.g.
/// if you want to see the RUB, LocB and the nodes that have been merged
#[derive(Debug, Builder)]
pub struct VizConfig {
    /// This flag must be true (default) if you want to see the value of
    /// each node (length of the longest path)
    #[builder(default = "true")]
    pub show_value: bool,
    /// This flag must be true (default) if you want to see the locb of
    /// each node (length of the longest path from the bottom)
    #[builder(default = "true")]
    pub show_locb: bool,
    /// This flag must be true (default) if you want to see the rub of
    /// each node (fast upper bound)
    #[builder(default = "true")]
    pub show_rub: bool,
    /// This flag must be true (default) if you want to see the threshold
    /// associated to the exact nodes
    #[builder(default = "true")]
    pub show_threshold: bool,
    /// This flag must be true (default) if you want to see all nodes that
    /// have been deleted because of restrict or relax operations
    #[builder(default = "false")]
    pub show_deleted: bool,
    /// This flag must be true (default) if you want to see the nodes that
    /// have been merged be grouped together (only applicable is show_deleted = true)
    #[builder(default = "false")]
    pub group_merged: bool,
}

impl<T, const CUTSET_TYPE: CutsetType> VectorMdd<T, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone,
{
    /// This is the method you will want to use in order to create the output image you would like.
    /// Note: the output is going to be a string of (not compiled) 'dot'. This makes it easier for
    /// me to code and gives you the freedom to fiddle with the graph if needed.
    pub fn as_graphviz(&self, config: &VizConfig) -> String {
        let mut out = String::new();

        out.push_str("digraph {\n\tranksep = 3;\n\n");

        // Show all nodes
        for (layer_id, layer) in self.nodes.iter().enumerate() {
            for (node_id, node) in layer.iter().enumerate() {
                if !config.show_deleted && node.flags.is_deleted() {
                    continue;
                }
                let id = NodeId(layer_id, node_id);
                out.push_str(&self.node(id, config));
                out.push_str(&self.edges_of(id));
            }
        }

        // Show clusters if requested
        if config.show_deleted && config.group_merged {
            for (layer_id, layer) in self.nodes.iter().enumerate() {
                let mut merged = Vec::with_capacity(layer.len());
                for (node_id, node) in layer.iter().enumerate() {
                    let id = NodeId(layer_id, node_id);
                    if node.flags.is_deleted() || node.flags.is_relaxed() {
                        merged.push(format!("node_{}_{}", id.0, id.1));
                    }
                }
                if !merged.is_empty() {
                    out.push_str(&format!("\tsubgraph cluster_{layer_id} "));
                    out.push_str("{\n");
                    out.push_str("\t\tstyle=filled;\n");
                    out.push_str("\t\tcolor=purple;\n");
                    out.push_str(&format!("\t\t{}\n", merged.join(";")));
                    out.push_str("\t};\n");
                }
            }
        }

        // Finish the graph with a terminal node
        out.push_str(&self.add_terminal_node());

        out.push_str("}\n");
        out
    }

    /// Creates a string representation of one single node
    fn node(&self, id: NodeId, config: &VizConfig) -> String {
        let attributes = self.node_attributes(id, config);
        format!("\tnode_{}_{} [{attributes}];\n", id.0, id.1)
    }

    #[allow(clippy::redundant_closure_call)]
    /// Creates a string representation of the edges incident to one node
    fn edges_of(&self, id: NodeId) -> String {
        let mut out = String::new();
        foreachincoming!(edge of id, self, |edge: Edge<T>| {
            let Edge{from, to, decision, cost,..} = &edge; //TODO is this clone expensive?
            let best = get!(node id, self).best;
            out.push_str(&Self::edge(from, to, decision, *cost, best.map_or(false, |eid| *get!(edge eid, self) == edge)));
        });
        out
    }
    /// Adds a terminal node (if the DD is feasible) and draws the edges entering that node from
    /// all the nodes of the terminal layer.
    fn add_terminal_node(&self) -> String {
        let mut out = String::new();
        let last_layer_id = self.nodes.len() - 1;
        let layer = get!(layer LayerId(last_layer_id), self);
        if !layer.is_empty() {
            let terminal = "\tterminal [shape=\"circle\", label=\"\", style=\"filled\", color=\"black\", group=\"terminal\"];\n";
            out.push_str(terminal);

            let terminal = layer;
            let vmax = terminal
                .iter()
                .map(|n| n.value_top)
                .max()
                .unwrap_or(isize::MAX);
            for (id, term) in terminal.iter().enumerate() {
                let value = term.value_top;
                if value == vmax {
                    out.push_str(&format!(
                        "\tnode_{}_{} -> terminal [penwidth=3];\n",
                        last_layer_id, id
                    ));
                } else {
                    out.push_str(&format!("\tnode_{}_{} -> terminal;\n", last_layer_id, id));
                }
            }
        }
        out
    }
    /// Creates a string representation of one edge
    fn edge(from: &NodeId, to: &NodeId, decision: &Decision, cost: isize, is_best: bool) -> String {
        let width = if is_best { 3 } else { 1 };
        let variable = decision.variable.0;
        let value = decision.value;
        let label = format!("(x{variable} = {value})\\ncost = {cost}");

        format!(
            "\tnode_{}_{} -> node_{}_{} [penwidth={width},label=\"{label}\"];\n",
            from.0, from.1, to.0, to.1
        )
    }
    /// Creates the list of attributes that are used to configure one node
    fn node_attributes(&self, id: NodeId, config: &VizConfig) -> String {
        let node = get!(node id, self);
        let merged = node.flags.is_relaxed();
        let state = node.state.as_ref();
        let restricted = node.flags.is_deleted();

        let shape = Self::node_shape(merged, restricted);
        let color = Self::node_color(node, merged);
        let peripheries = Self::node_peripheries(node);
        let group = self.node_group(node);
        let label = Self::node_label(node, id, state, config);

        format!("shape={shape},style=filled,color={color},peripheries={peripheries},group=\"{group}\",label=\"{label}\"")
    }
    /// Determines the group of a node based on the last branching decision leading to it
    fn node_group(&self, node: &Node<T>) -> String {
        if let Some(eid) = node.best {
            let edge = self.edges[eid.0].clone();
            format!("{}", edge.decision.variable.0)
        } else {
            "root".to_string()
        }
    }
    /// Determines the shape to use when displaying a node
    fn node_shape(merged: bool, restricted: bool) -> &'static str {
        if merged || restricted {
            "square"
        } else {
            "circle"
        }
    }
    /// Determines the number of peripheries to draw when displaying a node.
    fn node_peripheries(node: &Node<T>) -> usize {
        if node.flags.is_cutset() {
            4
        } else {
            1
        }
    }
    /// Determines the color of peripheries to draw when displaying a node.
    fn node_color(node: &Node<T>, merged: bool) -> &str {
        if node.flags.is_cutset() {
            "red"
        } else if node.flags.is_exact() {
            "\"#99ccff\""
        } else if merged {
            "yellow"
        } else {
            "lightgray"
        }
    }
    /// Creates text label to place inside of the node when displaying it
    fn node_label(node: &Node<T>, id: NodeId, _state: &T, config: &VizConfig) -> String {
        let mut out = format!("id: ({},{})", id.0, id.1);
        // out.push_str(&format!("\n{state:?}"));

        // let mut out = format!("{state:?}");

        if config.show_value {
            out.push_str(&format!("\\nval: {}", node.value_top));
        }
        if config.show_locb {
            out.push_str(&format!("\\nlocb: {}", Self::extreme(node.value_bot)));
        }
        if config.show_rub {
            out.push_str(&format!("\\nrub: {}", Self::extreme(node.rub)));
        }
        if config.show_threshold {
            out.push_str(&format!(
                "\\ntheta: {}",
                Self::extreme(node.theta.unwrap_or(isize::MAX))
            ));
        }

        out
    }
    /// An utility method to replace extreme values with +inf and -inf
    fn extreme(x: isize) -> String {
        match x {
            isize::MAX => "+inf".to_string(),
            isize::MIN => "-inf".to_string(),
            _ => format!("{x}"),
        }
    }
}
