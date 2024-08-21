//! This implemets the same decison diagrams as in clean and pooled with the difference being in the
//! data structures that represent the nodes and edges. Strict vectors for nodes and edges
//! are replaced with maps.
//!

//! This is an adaptation of the vector based architecture which implements all
//! the pruning techniques that I have proposed in my PhD thesis (RUB, LocB, EBPO).
//! It also implements the techniques we proposed in
//!
//! ``Decision Diagram-Based Branch-and-Bound with Caching
//! for Dominance and Suboptimality Detection''.

use std::{collections::{hash_map::Entry, HashSet}, fmt::Debug, hash::Hash, sync::Arc, fs};

use derive_builder::Builder;
use fxhash::FxHashMap;

use crate::{
    CompilationInput, CompilationType, Completion, CutsetType, Variable, Decision, DecisionDiagram,
    DominanceCheckResult, NodeFlags, Problem, Reason, Solution, SubProblem, FRONTIER,
    LAST_EXACT_LAYER,
};

/// The identifier of a node: it indicates the position of the referenced node
/// in the ’nodes’ vector (layer, index) of the mdd structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct NodeId(usize, usize);

/// The identifier of an edge: it indicates the position of the referenced edge
/// in the ’edges’ vector of the mdd structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct EdgeId(usize);

/// The identifier of an edge list: it indicates the position of an edge list
/// in the ’edgelists’ vector of the mdd structure.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct EdgesListId(usize);

/// The identifier of a layer: it indicates the position of the referenced layer
/// in the 'layers' vector of the mdd structure.
#[derive(Debug, Clone, Copy)]
struct LayerId(usize);

/// Represents an effective node from the decision diagram
#[derive(Debug, Clone)]
struct Node<T> {
    /// The state associated to this node
    state: Arc<T>,
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
    /// The identifier of the latest edge having been added to the input adjacency
    /// list of this node. (Edges, by themselves form a kind of linked structure)
    inbound: EdgesListId,
    /// The identifier of the latest edge having been added to the output adjacency
    /// list of this node. (Edges, by themselves form a kind of linked structure)
    outbound: EdgesListId,
    // The rough upper bound associated to this node
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
    /// List of edges that appear on some path to the node
    some: HashSet<(Variable,isize)>,
    /// List of edges that appear on all paths to the node
    all: HashSet<(Variable,isize)>,
}

/// Materializes one edge a.k.a arc from the decision diagram. It logically
/// connects two nodes and annotates the link with a decision and a cost.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Edge<T> {
    /// The identifier of the node at the ∗∗source∗∗ of this edge.
    from: NodeId,
    /// The identifier of the node at the ∗∗destination∗∗ of this edge.
    to: NodeId,
    /// This is the decision label associated to this edge. It gives the
    /// information "what variable" is assigned to "what value".
    decision: Arc<Decision<T>>,
    /// This is the transition cost of making this decision from the state
    /// associated with the source node of this edge.
    cost: isize,
}

/// Represents a 'node' in the linked list that forms the adjacent edges list for a node
#[derive(Debug, Clone, Copy, PartialEq)]
enum EdgesList {
    Cons { head: EdgeId, tail: EdgesListId },
    Nil,
}

struct EdgesListIter<'a> {
    content: &'a Vec<EdgesList>,
    current: EdgesList,
}

impl EdgesList {
    fn iter(self, vector: &Vec<EdgesList>) -> EdgesListIter {
        //non-consuming?
        EdgesListIter {
            content: vector,
            current: self,
        }
    }
}

impl<'a> Iterator for EdgesListIter<'a> {
    type Item = EdgesList;
    fn next(&mut self) -> Option<EdgesList> {
        match self.current {
            EdgesList::Nil => return None,
            EdgesList::Cons { head, tail } => {
                self.current = self.content[tail.0];
                return Some(EdgesList::Cons { head, tail });
            }
        }
    }
}

/// Represents a 'layer' in the decision diagram
#[derive(Debug, Clone, Copy)]
struct Layer {
    from: usize,
    to: usize,
    size: usize,
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
pub struct Mapped<T, X, const CUTSET_TYPE: CutsetType>
where
    T: Eq + PartialEq + Hash + Clone,
    X: Eq + PartialEq + Hash + Clone,
{
    /// All the nodes composing this decision diagram. The vector comprises
    /// nodes from all layers in the DD. A nice property is that all nodes
    /// belonging to one same layer form a sequence in the ‘nodes‘ vector.
    nodes: Vec<Vec<Node<T>>>,
    /// This vector stores the information about all edges connecting the nodes
    /// of the decision diagram.
    edges: Vec<Edge<X>>,
    /// This vector stores the information about all incoming edge lists constituting
    /// linked lists between edges
    in_edgelists: Vec<EdgesList>,
    /// This vector stores the information about all outgoing edge lists constituting
    /// linked lists between edges
    out_edgelists: Vec<EdgesList>,

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
    /// The depth of the layer currently being expanded
    curr_depth: usize,

    /// Keeps track of the decisions that have been taken to reach the root
    /// of this DD, starting from the problem root.
    path_to_root: Vec<Arc<Decision<X>>>,
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

const NIL: EdgesListId = EdgesListId(0);

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
    (    in_edgelist $id:expr, $dd:expr) => {
        &$dd.in_edgelists[$id.0]
    };
    (mut in_edgelist $id:expr, $dd:expr) => {
        &mut $dd.in_edgelists[$id.0]
    };
    (    out_edgelist $id:expr, $dd:expr) => {
        &$dd.out_edgelists[$id.0]
    };
    (mut out_edgelist $id:expr, $dd:expr) => {
        &mut $dd.out_edgelists[$id.0]
    };
    (    layer    $id:expr, $dd:expr) => {
        &$dd.nodes[$id.0]
    };
    (mut layer    $id:expr, $dd:expr) => {
        &mut $dd.nodes[$id.0]
    };
}

/// This macro performs an action for each incoming edge of a given node in the dd
/// // TODO: Confirm functionality here, I cloned edge instead of the copy via derefenecing it was doing before.
/// // How does that change behaviour?
macro_rules! foreach {
    (edge of $id:expr, $dd:expr, $action:expr) => {
        let mut list = get!(node $id, $dd).inbound;
        while let EdgesList::Cons{head, tail} = *get!(in_edgelist list, $dd) {
            let edge = get!(edge head, $dd).clone();
            $action(edge);
            list = tail;
        }
    };
}

/// This macro appends an edge to the list of edges adjacent to a given node
macro_rules! append_edge_to {
    ($dd:expr, $edge:expr) => {
        let new_eid = EdgeId($dd.edges.len());
        let lst_id  = EdgesListId($dd.in_edgelists.len());
        $dd.edges.push($edge);
        $dd.in_edgelists.push(EdgesList::Cons { head: new_eid, tail: get!(node $edge.to, $dd).inbound });
        $dd.out_edgelists.push(EdgesList::Cons { head: new_eid, tail: get!(node $edge.from, $dd).outbound });

        let parent = get!(mut node $edge.from, $dd);
        let parent_exact = parent.flags.is_exact();
        let value = parent.value_top.saturating_add($edge.cost);
        parent.outbound = lst_id;

        let node = get!(mut node $edge.to, $dd);
        let exact = parent_exact & node.flags.is_exact();
        node.flags.set_exact(exact);
        node.inbound = lst_id;

        if value >= node.value_top {
            node.best = Some(new_eid);
            node.value_top = value;
        }
    };
}

/// This macro detaches an edge to the list of edges adjacent to a given node
//TODO verify functionality
macro_rules! detach_edge_from {
    ($dd:expr, $node_id:expr, $edgeslist:expr) => {
        let node = get!(mut node $node_id, $dd);
        let mut prev_id = node.inbound;
        let mut curr_id = node.inbound;

        // detach from inbound list
        let mut list_end:bool  = false;
        while !list_end{
            let curr = get!(in_edgelist curr_id, $dd);
            match curr{
                EdgesList::Cons{head:_curr_head, tail:curr_tail} => {

                            if *curr == $edgeslist  {
                                //TODO  disconnect curr by making tail NIL : currently breaks on some mutability P but likely can live with it cause noone points to curr anymore
                                // {$dd.in_edgelists[curr_id.0] =  EdgesList::Cons{head:*curr_head,tail:NIL};}

                                if curr_id == node.inbound {// if i am at start
                                    node.inbound = *curr_tail;
                                    // $dd.in_edgelists[curr_id.0] = EdgesList::Cons{head:*curr_head,tail:NIL};
                                }
                                else if *curr_tail == NIL{ // if i am at the end
                                    node.inbound = NIL;
                                }
                                else {// if i am in the middle
                                    // $dd.in_edgelists[curr_id.0] =  EdgesList::Cons{head:*curr_head,tail:NIL};
                                    // *curr_tail = NIL;
                                    let prev = get!(in_edgelist prev_id, $dd);
                                    match prev {
                                        EdgesList::Cons{head:prev_head, tail:_prev_tail} => $dd.in_edgelists[prev_id.0] = EdgesList::Cons{head:*prev_head,tail:*curr_tail},
                                        _ => {}
                                    }
                                }
                                break;
                            }
                            prev_id = curr_id;
                            curr_id = *curr_tail;
                        },
                EdgesList::Nil => {
                                list_end = true;
                    }
            }
        }


        // detach from outbound list
        list_end  = false;
        while !list_end{
            let curr = get!(out_edgelist curr_id, $dd);
            match curr{
                EdgesList::Cons{head:_curr_head, tail:curr_tail} => {

                            if *curr == $edgeslist  {
                                //TODO  disconnect curr by making tail NIL : currently breaks on some mutability P but likely can live with it cause noone points to curr anymore
                                // {$dd.in_edgelists[curr_id.0] =  EdgesList::Cons{head:*curr_head,tail:NIL};}

                                if curr_id == node.outbound {// if i am at start
                                    node.outbound = *curr_tail;
                                    // $dd.in_edgelists[curr_id.0] = EdgesList::Cons{head:*curr_head,tail:NIL};
                                }
                                else if *curr_tail == NIL{ // if i am at the end
                                    node.outbound = NIL;
                                }
                                else {// if i am in the middle
                                    // $dd.in_edgelists[curr_id.0] =  EdgesList::Cons{head:*curr_head,tail:NIL};
                                    // *curr_tail = NIL;
                                    let prev = get!(out_edgelist prev_id, $dd);
                                    match prev {
                                        EdgesList::Cons{head:prev_head, tail:_prev_tail} => $dd.out_edgelists[prev_id.0] = EdgesList::Cons{head:*prev_head,tail:*curr_tail},
                                        _ => {}
                                    }
                                }
                                break;
                            }
                            prev_id = curr_id;
                            curr_id = *curr_tail;
                        },
                EdgesList::Nil => {
                                list_end = true;
                    }
            }
        }
    };
}

impl<T, X, const CUTSET_TYPE: CutsetType> Default for Mapped<T, X, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone + Debug,
    X: Eq + PartialEq + Hash + Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, X, const CUTSET_TYPE: CutsetType> DecisionDiagram for Mapped<T, X, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone + Debug,
    X: Eq + PartialEq + Hash + Clone + Debug,
{
    type State = T;
    type DecisionState = X;

    fn compile(
        &mut self,
        input: &CompilationInput<Self::State, Self::DecisionState>,
    ) -> Result<Completion, Reason> {
        self._compile(input)
    }

    fn refine(
        &mut self,
        input: &CompilationInput<Self::State, Self::DecisionState>,
    ) -> Result<Completion, Reason> {
        self._refine(input)
    }

    fn is_exact(&self) -> bool {
        self.is_exact || self.has_exact_best_path
    }

    fn best_value(&self) -> Option<isize> {
        self._best_value()
    }

    fn best_solution(&self) -> Option<Solution<X>> {
        self._best_solution()
    }

    fn best_exact_value(&self) -> Option<isize> {
        self._best_exact_value()
    }

    fn best_exact_solution(&self) -> Option<Solution<X>> {
        self._best_exact_solution()
    }

    fn drain_cutset<F>(&mut self, func: F)
    where
        F: FnMut(SubProblem<Self::State, Self::DecisionState>),
    {
        self._drain_cutset(func)
    }
}

impl<T, X, const CUTSET_TYPE: CutsetType> Mapped<T, X, { CUTSET_TYPE }>
where
    T: Eq + PartialEq + Hash + Clone + Debug,
    X: Eq + PartialEq + Hash + Clone + Debug,
{
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            in_edgelists: vec![],
            out_edgelists: vec![],
            //
            prev_l: vec![],
            next_l: Default::default(),
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
        // do not reset lel, only start expanding from last point of lel to reduce redundant work
        self.best_node = None;
        self.best_exact_node = None;
        self.is_exact = false;
        self.has_exact_best_path = false;
    }

    fn _clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.in_edgelists.clear();
        self.out_edgelists.clear();
        self.prev_l.clear();
        self.next_l.clear();
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

    fn _best_solution(&self) -> Option<Vec<Arc<Decision<X>>>> {
        self.best_node.map(|id| self._best_path(id))
    }

    fn _best_exact_value(&self) -> Option<isize> {
        self.best_exact_node.map(|id| get!(node id, self).value_top)
    }

    fn _best_exact_solution(&self) -> Option<Vec<Arc<Decision<X>>>> {
        self.best_exact_node.map(|id| self._best_path(id))
    }

    fn _best_path(&self, id: NodeId) -> Vec<Arc<Decision<X>>> {
        Self::_best_path_partial_borrow(id, &self.path_to_root, &self.nodes, &self.edges)
    }

    fn _best_path_partial_borrow(
        id: NodeId,
        root_pa: &[Arc<Decision<X>>],
        nodes: &[Vec<Node<T>>],
        edges: &[Edge<X>],
    ) -> Vec<Arc<Decision<X>>> {
        let mut sol = root_pa.to_owned();
        let mut edge_id = nodes[id.0][id.1].best;
        while let Some(eid) = edge_id {
            let edge = edges[eid.0].clone();
            println!(
                "finding best path with edge from {:?} to {:?} with cost {:?}",
                edge.from, edge.to, edge.cost
            );
            sol.push(edge.decision);
            edge_id = nodes[edge.from.0][edge.from.1].best;
        }
        sol
    }

    fn _compile(&mut self, input: &CompilationInput<T, X>) -> Result<Completion, Reason> {
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
                let state = get!(node node_id, self).state.clone();
                let rub = input.relaxation.fast_upper_bound(state.as_ref());
                get!(mut node node_id, self).rub = rub;
                let ub = rub.saturating_add(get!(node node_id, self).value_top);
                if ub > input.best_lb {
                    input
                        .problem
                        .for_each_in_domain(var, state.as_ref(), &mut |decision| {
                            self._branch_on(*node_id, decision, input.problem)
                        })
                }
            }

            self.curr_depth += 1;
        }

        self._finalize(input);

        Ok(Completion {
            is_exact: self.is_exact(),
            best_value: self.best_node.map(|n| get!(node n, self).value_top),
        })
    }

    fn _refine(&mut self, input: &CompilationInput<T, X>) -> Result<Completion, Reason> {
        // clear parts of diagram reset by refinement
        self._clear_for_refine();

        //visualise for debug
        let mut config = VizConfigBuilder::default().build().unwrap();
        // config.show_deleted = true;
        config.group_merged = true;
        print!("before refine\n");
        let s = self.as_graphviz(&config);
        fs::write("/home/eaeigbe/Documents/PhD/ddo/resources/visualisation_tests/incremental.dot", s).expect("Unable to write file");

        // intialise diagram but without emptying nodes and edges etc
        // only enough that we start analysis from top to bottom
        self.path_to_root.extend_from_slice(&input.residual.path);
        self.curr_depth = input.residual.depth;
        // let root_node_id = NodeId(0);
        self.next_l.clear();
        // self.next_l.insert(input.residual.state.clone(), root_node_id);

        // go layer by layer
        // is this always ordered that way? we need to do top to bottom traversal
        let mut curr_layer_id = self.lel.0;

        while curr_layer_id < self.nodes.len() {
            if input.cutoff.must_stop() {
                return Err(Reason::CutoffOccurred);
            }
            // print!("refining layer {curr_layer_id} \n");
            if !self._refine_curr_layer(input, curr_layer_id) {
                break;
            }

            // update node parameters?

            curr_layer_id += 1;
            self.curr_depth += 1;
        }

        //TODO DO this betterr!
        // populate next_l as last layer
        // fetch final layer nodes
        let final_layer_id = LayerId(self.nodes.len() - 1);
        let final_layer = get!(layer final_layer_id, self);

        //TODO only collect those that are not deleted
        let print_final_l = final_layer
            .iter().enumerate()
            .filter_map(|(node_id,node)| (!node.flags.is_deleted()).then_some((node_id,node)))
            .collect::<Vec<_>>();
        println!("Final layer {:?} contents while last exact layer is {:?}",final_layer_id,self.lel);
        for (node_id,node) in &print_final_l {
            println!("node {:?} is exact {:?} and is deleted {:?} with value {:?} ",NodeId(self.nodes.len() - 1,*node_id), 
                                        node.flags.is_exact(),node.flags.is_deleted(),node.value_top);
        }

        for (id, node) in final_layer.iter().enumerate() {
            if !node.flags.is_deleted(){
                self.next_l
                .insert(node.state.clone(), NodeId(final_layer_id.0, id));
            }
        }

        self._finalize_for_refine(input);

        print!("after refine\n");
        let s = self.as_graphviz(&config);
        fs::write("/home/eaeigbe/Documents/PhD/ddo/resources/visualisation_tests/incremental.dot", s).expect("Unable to write file");


        Ok(Completion {
            is_exact: self.is_exact(),
            best_value: self.best_node.map(|n| get!(node n, self).value_top),
        })
    }

    fn _initialize(&mut self, input: &CompilationInput<T, X>) {
        self.path_to_root.extend_from_slice(&input.residual.path);
        self.in_edgelists.push(EdgesList::Nil);
        self.out_edgelists.push(EdgesList::Nil);

        let root_node_id = NodeId(0, 0);
        // because I expect subproblems to begin with exact nodes, i know that some == all 
        // and all decisions in the path to root also are in both sets
        let some = self.path_to_root.iter().map(|x| (x.variable,x.value)).collect();
        let all = self.path_to_root.iter().map(|x| (x.variable,x.value)).collect();
        
        let root_node = Node {
            state: input.residual.state.clone(),
            value_top: input.residual.value,
            value_bot: isize::MIN,
            best: None,
            inbound: NIL,
            outbound: NIL,
            rub: isize::MAX,
            theta: None,
            flags: NodeFlags::new_exact(),
            depth: input.residual.depth,
            some: some,
            all: all,
        };

        self.nodes.push(vec![root_node]);
        self.next_l
            .insert(input.residual.state.clone(), root_node_id);
        //TODO why are we pushing Nil twice? Line 507 above and here too
        self.in_edgelists.push(EdgesList::Nil);
        self.out_edgelists.push(EdgesList::Nil);
        self.curr_depth = input.residual.depth;
    }

    fn _finalize(&mut self, input: &CompilationInput<T, X>) {
        self._find_best_node();
        self._finalize_exact(input);
        self._finalize_cutset(input);
        self._compute_local_bounds(input);
        self._compute_thresholds(input);
    }

    fn _finalize_for_refine(&mut self, input: &CompilationInput<T, X>) {
        self._find_best_node();
        println!("found best node");
        self._finalize_exact(input);
        self._finalize_cutset(input);
        self._compute_local_bounds(input);
        self._compute_thresholds(input);
        println!("completed finalize");
    }

    fn _drain_cutset<F>(&mut self, mut func: F)
    where
        F: FnMut(SubProblem<T, X>),
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
    fn _compute_local_bounds(&mut self, input: &CompilationInput<T, X>) {
        //FIXME ?? how does this work -- first part of check always true
        if self.lel.0 < self.nodes.len() && input.comp_type == CompilationType::Relaxed {
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
                        foreach!(edge of id, self, |edge: Edge<X>| {
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
    fn _compute_thresholds(&mut self, input: &CompilationInput<T, X>) {
        if input.comp_type == CompilationType::Relaxed || self.is_exact {
            let mut best_known = input.best_lb;

            if let Some(best_exact_node) = self.best_exact_node {
                let best_exact_value = get!(mut node best_exact_node, self).value_top;
                best_known = best_known.max(best_exact_value);

                for id in self.next_l.values() {
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
                        foreach!(edge of id, self, |edge: Edge<X>| {
                            let parent = get!(mut node edge.from, self);
                            let theta  = parent.theta.unwrap_or(isize::MAX);
                            parent.theta = Some(theta.min(my_theta.saturating_sub(edge.cost)));
                        });
                    }
                }
            }
        }
    }

    fn _maybe_update_cache(node: &Node<T>, input: &CompilationInput<T, X>) {
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

    fn _finalize_cutset(&mut self, input: &CompilationInput<T, X>) {
        // if self.lel.is_none() {
        //     self.lel = Some(LayerId(self.nodes.len())); // all nodes of the DD are above cutset
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
        if lel.0 < self.nodes.len() {
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
                    foreach!(edge of id, self, |edge: Edge<X>| {
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

    fn _find_best_node(&mut self) {
        self.best_node = self
            .next_l
            .values()
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
        if let Some(x) = self.best_node {
            println!(
                "best node {:?} is exact {:?} at value {:?}",
                x,
                get!(node x, self).flags.is_exact(),
                get!(node x, self).value_top
            );
        };
        self.best_exact_node = self
            .next_l
            .values()
            .filter(|id| get!(node id, self).flags.is_exact())
            .copied()
            .max_by_key(|id| get!(node id, self).value_top);
        if let Some(x) = self.best_exact_node {
            println!(
                "best exact node {:?} is exact {:?} at value {:?}",
                x,
                get!(node x, self).flags.is_exact(),
                get!(node x, self).value_top
            );
        };

    }

    fn _finalize_exact(&mut self, input: &CompilationInput<T, X>) {
        self.is_exact = self.lel.0 == self.nodes.len() - 1;
        if self.is_exact{
            println!()
        }
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

    fn _refine_curr_layer(&mut self, input: &CompilationInput<T, X>, curr_layer_id: usize) -> bool {
        // fetch layer nodes
        // let prev_layer: Option<&Vec<Node<T>>> = {
        //     if curr_layer_id > 0 {
        //         Some(get!(layer LayerId(curr_layer_id-1),self))
        //     } else {
        //         None
        //     }
        // };
        let curr_layer = get!(mut layer LayerId(curr_layer_id), self);

        //TODO only collect those that are not deleted
        let mut curr_l = curr_layer
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
            if curr_layer_id == 0 {
                self._filter_with_cache(input, &mut curr_l);
            }
            self._filter_constraints(input, &mut curr_l);

            self._filter_with_dominance(input, &mut curr_l);

            self._stretch_if_needed(input, &mut curr_l, curr_layer_id);

            true
        }
    }

    fn _move_to_next_layer(
        &mut self,
        input: &CompilationInput<T, X>,
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
            if self.nodes.len() == 1 {
                self._filter_with_cache(input, curr_l);
            }
            self._filter_with_dominance(input, curr_l);

            self._squash_if_needed(input, curr_l);

            self.nodes.push(vec![]);
            true
        }
    }

    fn _filter_with_dominance(&mut self, input: &CompilationInput<T, X>, curr_l: &mut Vec<NodeId>) {
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

    fn _filter_with_cache(&mut self, input: &CompilationInput<T, X>, curr_l: &mut Vec<NodeId>) {
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
    fn _filter_constraints(&mut self, input: &CompilationInput<T, X>, curr_l: &mut Vec<NodeId>) {
        for node_id in curr_l.iter() {
            // let state = get!(node node_id, self).state.as_ref();
            let state = get!(mut node node_id, self).state.clone(); //TODO cloning to satisfy borrow checker, look into
            let outbound_decisions: Vec<(EdgesList, Arc<Decision<_>>)> = self.out_edgelists
                [get!(node node_id, self).outbound.0]
                .iter(&self.out_edgelists)
                .filter_map(|x| match x {
                    EdgesList::Cons { head, tail: _ } => {
                        Some((x, self.edges[head.0].decision.clone()))
                    }
                    _ => None,
                })
                .collect();
            for (e_list, e_dec) in outbound_decisions {
                if input.problem.filter(&state, &e_dec) {
                    // delete this edge
                    // detach_edge_from!(self, node_id, e_list);
                }
            }
        }
        //TODO if outbound edges all detached, set node as deleted-- basically if outbound is nil? I think
    }

    fn _branch_on(
        &mut self,
        from_id: NodeId,
        decision: Arc<Decision<X>>, //TODO why do I only take the type by refernece? matched definitions of functions transition and transiton_cost in dp.rs but why
        problem: &dyn Problem<State = T, DecisionState = X>,
    ) {
        let state = get!(node from_id, self).state.as_ref();
        let next_state = Arc::new(problem.transition(state, decision.as_ref()));
        let cost = problem.transition_cost(state, next_state.as_ref(), decision.as_ref());
        let next_layer_id = from_id.0 + 1;

        match self.next_l.entry(next_state.clone()) {
            Entry::Vacant(e) => {
                let parent = get!(node from_id, self).clone();
                let node_id = NodeId(next_layer_id, self.nodes[next_layer_id].len());
                let mut flags = NodeFlags::new_exact();
                flags.set_exact(parent.flags.is_exact());

                if self.nodes.len() == next_layer_id {
                    self.nodes.push(vec![]);
                }

                let mut some = parent.some.clone();
                some.insert((decision.variable,decision.value));

                // we can get away with insertint into the all set during branch because
                // we only branch one one deciison at a time so its defintely contained
                let mut all = parent.all.clone();
                all.insert((decision.variable,decision.value));

                self.nodes[next_layer_id].push(Node {
                    state: next_state,
                    value_top: parent.value_top.saturating_add(cost),
                    value_bot: isize::MIN,
                    //
                    best: None,
                    inbound: NIL,
                    outbound: NIL,
                    //
                    rub: isize::MAX,
                    theta: None,
                    flags,
                    depth: parent.depth + 1,
                    some: some,
                    all: all
                });
                append_edge_to!(
                    self,
                    Edge {
                        from: from_id,
                        to: node_id,
                        decision: decision.clone(),
                        cost,
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
                        decision: decision.clone(),
                        cost,
                    }
                );
            }
        }
    }

    fn _squash_if_needed(&mut self, input: &CompilationInput<T, X>, curr_l: &mut Vec<NodeId>) {
        if self.nodes.len() > 1 {
            match input.comp_type {
                CompilationType::Exact => { /* do nothing: you want to explore the complete DD */ }
                CompilationType::Restricted => {
                        if curr_l.len() > input.max_width{
                            self._restrict(input, curr_l)
                        }
                        else{
                            self._maybe_save_lel();
                        }
                }
                CompilationType::Relaxed => {
                    // TODO: self.nodes.len() is one ahead of self.layers.len() from clean.rs; why was it self.layers.len() > 1?
                    //       Because self.nodes.len() > 1 seems more correct, we don't stretch layer 0, but otherwise should be fine
                    if curr_l.len() > input.max_width {
                        self._relax(input, curr_l)
                    }
                    else {
                        self._maybe_save_lel();
                    }
                }
            }
        }
    }

    fn _stretch_if_needed(
        &mut self,
        input: &CompilationInput<T, X>,
        curr_l: &mut Vec<NodeId>,
        curr_layer_id: usize,
    ) {
        // let mut curr_layer = get!(mut layer LayerId(curr_layer_id), self);

        match input.comp_type {
            CompilationType::Restricted => { /* refinement does not build restrictions */ }

            _ => {
                /* for both exact and realaxed we just keep splitting and filtering */
                if curr_layer_id > 0 { /* we dont want to stretch the root of the subproblem */
                    let mut fully_split = false;
                    while !fully_split {
                        if curr_l.len() < input.max_width {
                            fully_split = self._split_layer(input, curr_l, curr_layer_id);
                        }
                        else {
                            break; // nothing to stretch
                        }
                    }
                    if fully_split && 
                        self.lel.0 == curr_layer_id -1 &&
                        curr_l.iter().filter(|x| !get!(node x, self).flags.is_deleted()).all(|x| get!(node x, self).flags.is_exact()){
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

    fn _restrict(&mut self, input: &CompilationInput<T, X>, curr_l: &mut Vec<NodeId>) {
        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| {
                    input.ranking.compare(
                        get!(node a, self).state.as_ref(),
                        get!(node b, self).state.as_ref(),
                    )
                })
                .reverse()
        }); // reverse because greater means more likely to be kept

        for drop_id in curr_l.iter().skip(input.max_width).copied() {
            get!(mut node drop_id, self).flags.set_deleted(true);
        }

        curr_l.truncate(input.max_width);
    }

    fn _update_some(&self,edge:&Edge<X>) -> HashSet<(Variable, isize)>{
        let mut some = get!(node edge.from, self).some.clone();
        assert!(!get!(node edge.from, self).flags.is_deleted(),"attempting to use a deleted parent node");
        some.insert((edge.decision.variable,edge.decision.value));
        some
    }

    fn _update_all(&self,edge:&Edge<X>) -> HashSet<(Variable, isize)>{
        let mut all = get!(node edge.from, self).all.clone();
        assert!(!get!(node edge.from, self).flags.is_deleted(),"attempting to use a deleted parent node");
        all.insert((edge.decision.variable,edge.decision.value));
        all
    }

    #[allow(clippy::redundant_closure_call)]
    fn _relax(&mut self, input: &CompilationInput<T, X>, curr_l: &mut Vec<NodeId>) {
        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| {
                    input.ranking.compare(
                        get!(node a, self).state.as_ref(),
                        get!(node b, self).state.as_ref(),
                    )
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
                best: None,    // yet
                inbound: NIL,  // yet
                outbound: NIL, // yet
                //
                rub: isize::MAX,
                theta: None,
                flags: NodeFlags::new_relaxed(),
                depth,
                some: HashSet::new(),
                all: HashSet::new(),
            });
            node_id
        });

        get!(mut node merged_id, self).flags.set_relaxed(true);

        let mut some: HashSet<(Variable,isize)> = HashSet::new();
        let mut all: HashSet<(Variable,isize)> = HashSet::new();

        for drop_id in merge {
            get!(mut node drop_id, self).flags.set_deleted(true);

            foreach!(edge of drop_id, self, |edge: Edge<X>| {
                if !get!(node edge.from, self).flags.is_deleted(){
                    let src   = get!(node edge.from, self).state.as_ref();
                    let dst   = get!(node edge.to,   self).state.as_ref();
                    let rcost = input.relaxation.relax(src, dst, merged.as_ref(), edge.decision.as_ref(), edge.cost);

                    append_edge_to!(self, Edge {
                        from: edge.from,
                        to: merged_id,
                        decision: edge.decision.clone(),
                        cost: rcost
                    });

                    some =  &some | &self._update_some(&edge);
                    if all.is_empty(){
                        all = self._update_all(&edge);
                    }
                    else {
                        all =  &all & &self._update_all(&edge);
                    }
                }
            });
        }

        get!(mut node merged_id, self).some = some;
        get!(mut node merged_id, self).all = all;

        if recycled.is_some() {
            curr_l.truncate(input.max_width);
            let saved_id = curr_l[input.max_width - 1];
            let flags = &mut get!(mut node saved_id, self).flags;
            flags.set_deleted(false);
        } else {
            curr_l.truncate(input.max_width - 1);
            curr_l.push(merged_id);
        }
    }

    fn _redirect_edges_after_split(&mut self,
        input: &CompilationInput<T, X>,
        split_states:&Vec<(Arc<T>,Vec<usize>)>,
        node_to_split_id: NodeId,
        curr_layer_id: LayerId,
        // inbound_edges: &Vec<&(usize, &Decision<X>)>,
        outbound_edges: &Vec<EdgeId>) -> Vec<NodeId>{
        let mut new_nodes = vec![];
        for (state, edges_to_append) in split_states {
            // only add merged as new node
            //TODO is this Id what i should be? If I insert, its id is different
            let curr_layer = get!(mut layer curr_layer_id, self);
            let split_id = NodeId(curr_layer_id.0, curr_layer.len());
            new_nodes.push(split_id);
            let depth = get!(node node_to_split_id, self).depth;
            let curr_layer = get!(mut layer curr_layer_id, self);
            curr_layer.push(Node {
                state: state.clone(),
                value_top: isize::MIN,
                value_bot: isize::MIN,
                best: None,    // yet
                inbound: NIL,  // yet
                outbound: NIL, // yet
                //
                rub: isize::MAX,
                theta: None,
                flags: NodeFlags::new_relaxed(),
                // flags: if edges_to_append.len() == 1 {NodeFlags::new_exact()} else {NodeFlags::new_relaxed()},
                depth,
                some: HashSet::new(),
                all: HashSet::new(),
            });

            
            let mut some: HashSet<(Variable,isize)> = HashSet::new();
            let mut all: HashSet<(Variable,isize)> = HashSet::new();
            //TODO redirect edges inbound
            let mut to_delete = true;
            for edge_id in edges_to_append {
                let e = self.edges[*edge_id].clone();
                if !get!(node e.from, self).flags.is_deleted() && !input.problem.filter(&get!(node e.from, self).state.clone(), &e.decision){
                    let cost = input.problem.transition_cost(
                        get!(node e.from, self).state.as_ref(),
                        state.as_ref(),
                        e.decision.as_ref(), //TODO: update decision state?
                    );
                    append_edge_to!(
                        self,
                        Edge {
                            from: e.from,
                            to: split_id,
                            decision: e.decision.clone(), //TODO: update decision state?
                            cost: cost
                        }
                    ); 

                    some =  &some | &self._update_some(&e);
                    if all.is_empty(){
                        all = self._update_all(&e);
                    }
                    else {
                        all =  &all & &self._update_all(&e);
                    }
                    to_delete = false;
                }
            }
            // set node to exact if some and all and relaxed otherwise
            // exactness is further corrected by edge additon which check sparent exactness too
            // TODO: set correct exactness here
            if some == all {
                get!(mut node split_id, self).flags.set_exact(true);
            }
            else {
                get!(mut node split_id, self).flags.set_relaxed(true);
            }
            if to_delete{
                get!(mut node split_id, self).flags.set_deleted(true);
            }

            get!(mut node split_id, self).some = some;
            get!(mut node split_id, self).all = all;

            //TODO replicate all edges outbound - we need to recalculate the decision states because it changes upon split - also filter infeasible outbounds
            //TODO reserve size of outgoing
            if !get!(node split_id, self).flags.is_deleted(){
                let mut outgoing_nodes_to_update = Vec::with_capacity(outbound_edges.len());
                for edge_id in outbound_edges {
                    let e = self.edges[edge_id.0].clone();
                        if !get!(node e.to, self).flags.is_deleted() {
                            //TODO confirm if not filtering any edges leading to a node makes it safe to only recompute all for that node
                            outgoing_nodes_to_update.push(e.to);
                            let cost = input.problem.transition_cost(
                                state.as_ref(),
                                get!(node e.to, self).state.as_ref(),
                                e.decision.as_ref(),
                            );
                            println!("trying to add edge with variable {} assigned value {} to node ({},{})",
                                                    e.decision.variable.0,
                                                    e.decision.value,
                                                    split_id.0,split_id.1);
                            if !input.problem.filter(&state, &e.decision) {
                                println!("added edge with variable {} assigned value {} to node ({},{})",
                                                    e.decision.variable.0,
                                                    e.decision.value,
                                                    split_id.0,split_id.1);
                                append_edge_to!(
                                    self,
                                    Edge {
                                        from: split_id,
                                        to: e.to,
                                        decision: e.decision.clone(),
                                        cost: cost
                                    }
                                ); 
                            }
                        }
                }
            
            // update outgoing node edges
            for outgoing_node_id in outgoing_nodes_to_update{
                let mut some: HashSet<(Variable,isize)> = HashSet::new();
                let mut all: HashSet<(Variable,isize)> = HashSet::new();
                let inbound_start_for_outgoing = self.in_edgelists[get!(node outgoing_node_id,self).inbound.0];
                let inbound_edges = inbound_start_for_outgoing
                .iter(&self.in_edgelists)
                .filter_map(|x| match x {
                    EdgesList::Cons { head, tail: _ } => {
                        Some(head)
                    }
                    _ => None,
                })
                .collect::<Vec<EdgeId>>();

                //update node attributes
                let merged = self._merge_states_from_incoming_edges(input, &inbound_edges.iter().map(|x| x.0).collect());
                get!(mut node outgoing_node_id, self).state = merged.clone();
                // get!(mut node outgoing_node_id, self).value_top = isize::MIN;
                // get!(mut node outgoing_node_id, self).value_bot = isize::MIN;
                // get!(mut node outgoing_node_id, self).theta = None;
                // get!(mut node outgoing_node_id, self).best = None;

                for e_id in &inbound_edges{
                    let e = get!(edge e_id,self);
                    if !get!(node e.from, self).flags.is_deleted() {
                        some =  &some | &self._update_some(&e);
                        if all.is_empty(){
                            all = self._update_all(&e);
                        }
                        else {
                            all =  &all & &self._update_all(&e);
                        }
                    }
                }

                get!(mut node outgoing_node_id, self).some = some;
                get!(mut node outgoing_node_id, self).all = all;
            }
            }
        }
        // //TODO we should then delete all incoming and outgoing edges to the split node
        // // basically call this detach for all the edges coming into the node
        // for e_list in inbound_start.iter(&self.in_edgelists).collect::<Vec<_>>(){
        //     // detach_edge_from!(self,node_to_split_id,e_list);
        // }
        new_nodes
    }


    fn _split_layer(
        &mut self,
        input: &CompilationInput<T, X>,
        curr_l: &mut Vec<NodeId>,
        curr_layer_id: usize,
    ) -> bool {
        // order vec node by ranking
        //TODO assume same ranking as deciding to merge for now but make API for user to add specific ranking here
        curr_l.sort_unstable_by(|a, b| {
            get!(node a, self)
                .value_top
                .cmp(&get!(node b, self).value_top)
                .then_with(|| {
                    input.ranking.compare(
                        get!(node a, self).state.as_ref(),
                        get!(node b, self).state.as_ref(),
                    )
                })
                .reverse()
        }); // reverse because greater means more likely to be kept

        // select worst node and split
        let mut index = curr_l.len();
        while index > 0 {
            let node_to_split_id = curr_l[index - 1];
            let node_to_split = get!(node node_to_split_id, self);
            let inbound_start = self.in_edgelists[node_to_split.inbound.0];
            let outbound_start = self.out_edgelists[node_to_split.outbound.0];

            // collect inbound and outbound edges from linked list structure
            let inbound_edges = inbound_start
                .iter(&self.in_edgelists)
                .filter_map(|x| match x {
                    EdgesList::Cons { head, tail: _ } => {
                        Some((head.0, self.edges[head.0].decision.as_ref()))
                    }
                    _ => None,
                })
                .collect::<Vec<(usize, &Decision<X>)>>();
            let outbound_edges = outbound_start
                .iter(&self.out_edgelists)
                .filter_map(|x| match x {
                    EdgesList::Cons { head, tail: _ } => Some(head),
                    _ => None,
                })
                .collect::<Vec<EdgeId>>();

            if inbound_edges.len() > 1 {
                let split_states =
                    self._split_node(node_to_split_id, input, &mut inbound_edges.into_iter());
                

                //Delete split node
                get!(mut node node_to_split_id, self)
                    .flags
                    .set_deleted(true);

            
                let mut new_nodes = self._redirect_edges_after_split(input,&split_states,node_to_split_id,LayerId(curr_layer_id),&outbound_edges);
                

                curr_l.remove(index - 1);
                curr_l.append(&mut new_nodes);
                return false;
            } else {
                index -= 1;
                if inbound_edges.is_empty(){
                    get!(mut node node_to_split_id, self).flags.set_deleted(true);
                }
            }
        }
        let mut config = VizConfigBuilder::default().build().unwrap();
        // config.show_deleted = true;
        config.group_merged = true;
        print!("after split layer {curr_layer_id}\n");
        let s = self.as_graphviz(&config);
        fs::write("/home/eaeigbe/Documents/PhD/ddo/resources/visualisation_tests/incremental.dot", s).expect("Unable to write file");

        true
    }

    fn _merge_states_from_incoming_edges (
        &self,
        input: &CompilationInput<T, X>,
        inbound_edges: &Vec<usize>) -> Arc<T> {
        let mut new_states = vec![];
        for edge_id in inbound_edges {
            // create states for each edge transition
            let current_decision = self.edges[*edge_id].decision.as_ref();
            let parent_state = get!(node(self.edges[*edge_id].from), self).state.as_ref();
            let next_state = Arc::new(input.problem.transition(parent_state, current_decision));
            // let cost = input.problem.transition_cost(
            //     parent_state,
            //     next_state.as_ref(),
            //     current_decision,
            // ); //TODO actually use this cost post split
            new_states.push(next_state);
        }

        // merge them as they go to the same state actually
        let merged = Arc::new(
            input
                .relaxation
                .merge(&mut new_states.iter().map(|x| x.as_ref())),
        );
        merged
    }

    fn _split_node(
        &self,
        node_to_split_id: NodeId,
        input: &CompilationInput<T, X>,
        inbound_edges: &mut dyn Iterator<Item = (usize, &Decision<X>)>,
    ) -> Vec<(Arc<T>, Vec<usize>)> {
        let mut after_split: Vec<(Arc<T>, Vec<usize>)> = vec![]; // this usize is actually an edge id
                                                                 // let node_to_split = get!(mut node node_to_split_id, self);
        let split_state = get!(node node_to_split_id, self).state.as_ref();
        let split_state_edges = input.problem.split_state_edges(split_state, inbound_edges);

        // create n_split new nodes and redirect edges
        for cluster in &split_state_edges {
            let merged = self._merge_states_from_incoming_edges(input, cluster);
            after_split.push((merged, cluster.clone()));
        }
        after_split
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

impl<T, X, const CUTSET_TYPE: CutsetType> Mapped<T, X, { CUTSET_TYPE }>
where
    T: Debug + Eq + PartialEq + Hash + Clone,
    X: Debug + Eq + PartialEq + Hash + Clone,
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
                let mut merged = vec![];
                for (node_id, node) in layer.iter().enumerate() {
                    let id = NodeId(layer_id, node_id);
                    let node = get!(node id, self);
                    if node.flags.is_deleted() || node.flags.is_relaxed() {
                        merged.push(format!("node_{}_{}", id.0,id.1));
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
        foreach!(edge of id, self, |edge: Edge<X>| {
            let Edge{from, to, decision, cost} = edge.clone(); //TODO is this clone expensive?
            let best = get!(node id, self).best;
            let best = best.map(|eid| get!(edge eid, self).clone());
            out.push_str(&Self::edge(from, to, decision, cost, Some(edge) == best));
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
    fn edge(
        from: NodeId,
        to: NodeId,
        decision: Arc<Decision<X>>,
        cost: isize,
        is_best: bool,
    ) -> String {
        let width = if is_best { 3 } else { 1 };
        let variable = decision.variable.0;
        let value = decision.value;
        let label = format!("(x{variable} = {value})\\ncost = {cost}");

        format!("\tnode_{}_{} -> node_{}_{} [penwidth={width},label=\"{label}\"];\n",from.0,from.1,to.0,to.1)
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
    fn node_label(node: &Node<T>, id: NodeId, state: &T, config: &VizConfig) -> String {
        let mut out = format!("id: ({},{})", id.0,id.1);
        out.push_str(&format!("\n{state:?}"));

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

// ############################################################################
// #### TESTS #################################################################
// ############################################################################

#[cfg(test)]
mod test_default_mdd {
    use std::cmp::Ordering;
    use std::sync::Arc;

    use fxhash::FxHashMap;

    use crate::{
        Cache, Cutoff, DecisionCallback, EmptyCache, EmptyDominanceChecker, NoCutoff, Relaxation,
        SimpleCache, StateRanking, Threshold, Variable,
    };

    use super::{
        CompilationInput, CompilationType, Decision, DecisionDiagram, Mapped, Problem, Reason,
        SubProblem, VizConfigBuilder, FRONTIER, LAST_EXACT_LAYER,
    };

    type DefaultMDD<State, DecisionState> = DefaultMDDLEL<State, DecisionState>;
    type DefaultMDDLEL<State, DecisionState> = Mapped<State, DecisionState, { LAST_EXACT_LAYER }>;
    type DefaultMDDFC<State, DecisionState> = Mapped<State, DecisionState, { FRONTIER }>;

    #[test]
    fn by_default_the_mdd_type_is_exact() {
        let mdd = Mapped::<usize, usize, { LAST_EXACT_LAYER }>::new();

        assert!(mdd.is_exact());
    }

    #[test]
    fn root_remembers_the_pa_from_the_fringe_node() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let mut input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState {
                    depth: 1,
                    value: 42,
                }),
                value: 42,
                path: vec![Arc::new(Decision {
                    variable: Variable(0),
                    value: 42,
                    state: None,
                })],
                ub: isize::MAX,
                depth: 1,
            },
            cache: &cache,
            dominance: &dominance,
        };

        let mut mdd = DefaultMDD::new();
        assert!(mdd.compile(&input).is_ok());
        assert_eq!(
            mdd.path_to_root,
            vec![Arc::new(Decision {
                variable: Variable(0),
                value: 42,
                state: None
            })]
        );

        input.comp_type = CompilationType::Relaxed;
        assert!(mdd.compile(&input).is_ok());
        assert_eq!(
            mdd.path_to_root,
            vec![Arc::new(Decision {
                variable: Variable(0),
                value: 42,
                state: None
            })]
        );

        input.comp_type = CompilationType::Restricted;
        assert!(mdd.compile(&input).is_ok());
        assert_eq!(
            mdd.path_to_root,
            vec![Arc::new(Decision {
                variable: Variable(0),
                value: 42,
                state: None
            })]
        );
    }

    // In an exact setup, the dummy problem would be 3*3*3 = 9 large at the bottom level
    #[test]
    fn exact_completely_unrolls_the_mdd_no_matter_its_width() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();

        assert!(mdd.compile(&input).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value(), Some(6));
        assert_eq!(
            mdd.best_solution().unwrap(),
            vec![
                Arc::new(Decision {
                    variable: Variable(2),
                    value: 2,
                    state: None
                }),
                Arc::new(Decision {
                    variable: Variable(1),
                    value: 2,
                    state: None
                }),
                Arc::new(Decision {
                    variable: Variable(0),
                    value: 2,
                    state: None
                }),
            ]
        );
    }

    #[test]
    fn restricted_drops_the_less_interesting_nodes() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();

        assert!(mdd.compile(&input).is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value().unwrap(), 6);
        assert_eq!(
            mdd.best_solution().unwrap(),
            vec![
                Arc::new(Decision {
                    variable: Variable(2),
                    value: 2,
                    state: None
                }),
                Arc::new(Decision {
                    variable: Variable(1),
                    value: 2,
                    state: None
                }),
                Arc::new(Decision {
                    variable: Variable(0),
                    value: 2,
                    state: None
                }),
            ]
        );
    }

    #[test]
    fn exact_no_cutoff_completion_must_be_coherent_with_outcome() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);

        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact, mdd.is_exact());
        assert_eq!(completion.best_value, mdd.best_value());
    }
    #[test]
    fn restricted_no_cutoff_completion_must_be_coherent_with_outcome_() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);

        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact, mdd.is_exact());
        assert_eq!(completion.best_value, mdd.best_value());
    }
    #[test]
    fn relaxed_no_cutoff_completion_must_be_coherent_with_outcome() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);

        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.is_exact, mdd.is_exact());
        assert_eq!(completion.best_value, mdd.best_value());
    }

    #[derive(Debug, Clone, Copy)]
    struct CutoffAlways;
    impl Cutoff for CutoffAlways {
        fn must_stop(&self) -> bool {
            true
        }
    }
    #[test]
    fn exact_fails_with_cutoff_when_cutoff_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &CutoffAlways,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }

    #[test]
    fn restricted_fails_with_cutoff_when_cutoff_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &CutoffAlways,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }
    #[test]
    fn relaxed_fails_with_cutoff_when_cutoff_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &CutoffAlways,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_err());
        assert_eq!(Some(Reason::CutoffOccurred), result.err());
    }

    #[test]
    fn relaxed_merges_the_less_interesting_nodes() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);

        assert!(result.is_ok());
        assert!(mdd.best_solution().is_some());
        assert_eq!(mdd.best_value().unwrap(), 24);
        assert_eq!(
            mdd.best_solution().unwrap(),
            vec![
                Arc::new(Decision {
                    variable: Variable(2),
                    value: 2,
                    state: None
                }),
                Arc::new(Decision {
                    variable: Variable(1),
                    value: 0,
                    state: None
                }), // that's a relaxed edge
                Arc::new(Decision {
                    variable: Variable(0),
                    value: 2,
                    state: None
                }),
            ]
        );
    }

    #[test]
    fn relaxed_populates_the_cutset_and_will_not_squash_first_layer() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        let mut cutset = vec![];
        mdd.drain_cutset(|n| cutset.push(n));
        assert_eq!(cutset.len(), 3); // L1 was not squashed even though it was 3 wide
    }

    #[test]
    fn an_exact_mdd_must_be_exact() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_exact_as_long_as_no_merge_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 10,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact())
    }

    #[test]
    fn a_relaxed_mdd_is_not_exact_when_a_merge_occurred() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_exact_as_long_as_no_restriction_occurs() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 10,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact())
    }
    #[test]
    fn a_restricted_mdd_is_not_exact_when_a_restriction_occurred() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 1,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact())
    }
    #[test]
    fn when_the_problem_is_infeasible_there_is_no_solution() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyInfeasibleProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn when_the_problem_is_infeasible_there_is_no_best_value() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyInfeasibleProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_value().is_none())
    }
    #[test]
    fn exact_skips_node_with_an_ub_less_than_best_known_lb() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: 1000,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn relaxed_skips_node_with_an_ub_less_than_best_known_lb() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: 1000,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn restricted_skips_node_with_an_ub_less_than_best_known_lb() {
        let cache = EmptyCache::new();
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: 1000,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn exact_skips_nodes_with_a_value_less_than_known_threshold() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 0 }), 1, 0, true);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 1 }), 1, 1, true);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 2 }), 1, 2, true);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Exact,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn relaxed_skips_nodes_with_a_value_less_than_known_threshold() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 0 }), 1, 0, true);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 1 }), 1, 1, true);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 2 }), 1, 2, true);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }
    #[test]
    fn restricted_skips_nodes_with_a_value_less_than_known_threshold() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 0 }), 1, 0, true);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 1 }), 1, 1, true);
        cache.update_threshold(Arc::new(DummyState { depth: 1, value: 2 }), 1, 2, true);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());
        assert!(mdd.best_solution().is_none())
    }

    #[test]
    fn restricted_mdd_computes_thresholds_when_exact() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 10,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact());

        let expected = vec![
            (
                DummyState { depth: 0, value: 0 },
                Some(Threshold {
                    value: 0,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 0 },
                Some(Threshold {
                    value: 2,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 1 },
                Some(Threshold {
                    value: 2,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 2 },
                Some(Threshold {
                    value: 2,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 0 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 1 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 2 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 3 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 4 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 0 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 1 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 2 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 3 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 4 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 5 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 6 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    #[test]
    fn relaxed_mdd_computes_thresholds_when_exact() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 10,
            best_lb: isize::MIN,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact());

        let expected = vec![
            (
                DummyState { depth: 0, value: 0 },
                Some(Threshold {
                    value: 0,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 0 },
                Some(Threshold {
                    value: 2,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 1 },
                Some(Threshold {
                    value: 2,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 2 },
                Some(Threshold {
                    value: 2,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 0 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 1 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 2 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 3 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 4 },
                Some(Threshold {
                    value: 4,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 0 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 1 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 2 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 3 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 4 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 5 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 3, value: 6 },
                Some(Threshold {
                    value: 6,
                    explored: true,
                }),
            ),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    #[test]
    fn restricted_mdd_computes_thresholds_when_all_pruned() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Restricted,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 10,
            best_lb: 15,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact());

        let expected = vec![
            (
                DummyState { depth: 0, value: 0 },
                Some(Threshold {
                    value: 1,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 0 },
                Some(Threshold {
                    value: 3,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 1 },
                Some(Threshold {
                    value: 3,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 2 },
                Some(Threshold {
                    value: 3,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 0 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 1 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 2 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 3 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 4 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (DummyState { depth: 3, value: 0 }, None),
            (DummyState { depth: 3, value: 1 }, None),
            (DummyState { depth: 3, value: 2 }, None),
            (DummyState { depth: 3, value: 3 }, None),
            (DummyState { depth: 3, value: 4 }, None),
            (DummyState { depth: 3, value: 5 }, None),
            (DummyState { depth: 3, value: 6 }, None),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    #[test]
    fn relaxed_mdd_computes_thresholds_when_all_pruned() {
        let mut cache = SimpleCache::default();
        cache.initialize(&DummyProblem);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &DummyProblem,
            relaxation: &DummyRelax,
            ranking: &DummyRanking,
            cutoff: &NoCutoff,
            max_width: 10,
            best_lb: 15,
            residual: &SubProblem {
                state: Arc::new(DummyState { depth: 0, value: 0 }),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDD::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(mdd.is_exact());

        let expected = vec![
            (
                DummyState { depth: 0, value: 0 },
                Some(Threshold {
                    value: 1,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 0 },
                Some(Threshold {
                    value: 3,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 1 },
                Some(Threshold {
                    value: 3,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 1, value: 2 },
                Some(Threshold {
                    value: 3,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 0 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 1 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 2 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 3 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (
                DummyState { depth: 2, value: 4 },
                Some(Threshold {
                    value: 5,
                    explored: true,
                }),
            ),
            (DummyState { depth: 3, value: 0 }, None),
            (DummyState { depth: 3, value: 1 }, None),
            (DummyState { depth: 3, value: 2 }, None),
            (DummyState { depth: 3, value: 3 }, None),
            (DummyState { depth: 3, value: 4 }, None),
            (DummyState { depth: 3, value: 5 }, None),
            (DummyState { depth: 3, value: 6 }, None),
        ];

        for (state, threshold) in expected.iter().copied() {
            assert_eq!(threshold, cache.get_threshold(&state, state.depth));
        }
    }

    /// The example problem and relaxation for the local bounds should generate
    /// the following relaxed MDD in which the layer 'a','b' is the LEL.
    ///
    /// ```plain
    ///                      r
    ///                   /     \
    ///                10        7
    ///               /           |
    ///             a              b
    ///             |     +--------+-------+
    ///             |     |        |       |
    ///             2     3        6       5
    ///              \   /         |       |
    ///                M           e       f
    ///                |           |     /   \
    ///                4           0   1      2
    ///                |           |  /        \
    ///                g            h           i
    ///                |            |           |
    ///                0            0           0
    ///                +------------+-----------+
    ///                             t
    /// ```
    ///
    #[derive(Copy, Clone)]
    struct LocBoundsAndThresholdsExamplePb;
    impl Problem for LocBoundsAndThresholdsExamplePb {
        type State = char;
        type DecisionState = usize;
        fn nb_variables(&self) -> usize {
            4
        }
        fn initial_state(&self) -> char {
            'r'
        }
        fn initial_value(&self) -> isize {
            0
        }
        fn next_variable(
            &self,
            _: usize,
            next_layer: &mut dyn Iterator<Item = &Self::State>,
        ) -> Option<Variable> {
            match next_layer.next().copied().unwrap_or('z') {
                'r' => Some(Variable(0)),
                'a' => Some(Variable(1)),
                'b' => Some(Variable(1)),
                // c, d are merged into M
                'c' => Some(Variable(2)),
                'd' => Some(Variable(2)),
                'M' => Some(Variable(2)),
                'e' => Some(Variable(2)),
                'f' => Some(Variable(2)),
                'g' => Some(Variable(0)),
                'h' => Some(Variable(0)),
                'i' => Some(Variable(0)),
                _ => None,
            }
        }
        fn for_each_in_domain(
            &self,
            variable: Variable,
            state: &Self::State,
            f: &mut dyn DecisionCallback<Self::DecisionState>,
        ) {
            /* do nothing, just consider that all domains are empty */
            (match *state {
                'r' => vec![10, 7],
                'a' => vec![2],
                'b' => vec![3, 6, 5],
                // c, d are merged into M
                'M' => vec![4],
                'e' => vec![0],
                'f' => vec![1, 2],
                'g' => vec![0],
                'h' => vec![0],
                'i' => vec![0],
                _ => vec![],
            })
            .iter()
            .copied()
            .for_each(&mut |value| {
                f.apply(Arc::new(Decision {
                    variable,
                    value,
                    state: None,
                }))
            })
        }

        fn transition(&self, state: &char, d: &Decision<Self::DecisionState>) -> char {
            match (*state, d.value) {
                ('r', 10) => 'a',
                ('r', 7) => 'b',
                ('a', 2) => 'c', // merged into M
                ('b', 3) => 'd', // merged into M
                ('b', 6) => 'e',
                ('b', 5) => 'f',
                ('M', 4) => 'g',
                ('e', 0) => 'h',
                ('f', 1) => 'h',
                ('f', 2) => 'i',
                _ => 't',
            }
        }

        fn transition_cost(
            &self,
            _: &char,
            _: &Self::State,
            d: &Decision<Self::DecisionState>,
        ) -> isize {
            d.value
        }
    }

    #[derive(Copy, Clone)]
    struct LocBoundsAndThresholdsExampleRelax;
    impl Relaxation for LocBoundsAndThresholdsExampleRelax {
        type State = char;
        type DecisionState = usize;
        fn merge(&self, _: &mut dyn Iterator<Item = &char>) -> char {
            'M'
        }

        fn relax(
            &self,
            _: &char,
            _: &char,
            _: &char,
            _: &Decision<Self::DecisionState>,
            cost: isize,
        ) -> isize {
            cost
        }

        fn fast_upper_bound(&self, state: &char) -> isize {
            match *state {
                'r' => 30,
                'a' => 20,
                'b' => 20,
                // c, d are merged into M
                'M' => 10,
                'e' => 10,
                'f' => 10,
                'g' => 0,
                'h' => 0,
                'i' => 0,
                _ => 0,
            }
        }
    }

    #[derive(Clone, Copy)]
    struct CmpChar;
    impl StateRanking for CmpChar {
        type State = char;
        type DecisionState = usize;
        fn compare(&self, a: &char, b: &char) -> Ordering {
            a.cmp(b)
        }
    }

    #[test]
    fn relaxed_computes_local_bounds_and_thresholds_1() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 0,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDLEL::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact());
        assert_eq!(Some(16), mdd.best_value());

        let mut v = FxHashMap::<char, isize>::default();
        mdd.drain_cutset(|n| {
            v.insert(*n.state, n.ub);
        });

        assert_eq!(16, v[&'a']);
        assert_eq!(14, v[&'b']);
        assert_eq!(2, v.len());

        assert!(cache.get_threshold(&'r', 0).is_some());
        assert!(cache.get_threshold(&'a', 1).is_some());
        assert!(cache.get_threshold(&'b', 1).is_some());
        assert!(cache.get_threshold(&'M', 2).is_none());
        assert!(cache.get_threshold(&'e', 2).is_none());
        assert!(cache.get_threshold(&'f', 2).is_none());
        assert!(cache.get_threshold(&'g', 3).is_none());
        assert!(cache.get_threshold(&'h', 3).is_none());
        assert!(cache.get_threshold(&'i', 3).is_none());
        assert!(cache.get_threshold(&'t', 4).is_none());

        let mut threshold = cache.get_threshold(&'r', 0).unwrap();
        assert_eq!(0, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'a', 1).unwrap();
        assert_eq!(10, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'b', 1).unwrap();
        assert_eq!(7, threshold.value);
        assert!(!threshold.explored);
    }

    #[test]
    fn relaxed_computes_local_bounds_and_thresholds_2() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 0,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDFC::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact());
        assert_eq!(Some(16), mdd.best_value());

        let mut v = FxHashMap::<char, isize>::default();
        mdd.drain_cutset(|n| {
            v.insert(*n.state, n.ub);
        });

        assert_eq!(16, v[&'a']);
        assert_eq!(14, v[&'b']);
        assert_eq!(13, v[&'h']);
        assert_eq!(14, v[&'i']);
        assert_eq!(4, v.len());

        assert!(cache.get_threshold(&'r', 0).is_some());
        assert!(cache.get_threshold(&'a', 1).is_some());
        assert!(cache.get_threshold(&'b', 1).is_some());
        assert!(cache.get_threshold(&'M', 2).is_none());
        assert!(cache.get_threshold(&'e', 2).is_some());
        assert!(cache.get_threshold(&'f', 2).is_some());
        assert!(cache.get_threshold(&'g', 3).is_none());
        assert!(cache.get_threshold(&'h', 3).is_some());
        assert!(cache.get_threshold(&'i', 3).is_some());
        assert!(cache.get_threshold(&'t', 4).is_none());

        let mut threshold = cache.get_threshold(&'r', 0).unwrap();
        assert_eq!(0, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'a', 1).unwrap();
        assert_eq!(10, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'b', 1).unwrap();
        assert_eq!(7, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'e', 2).unwrap();
        assert_eq!(13, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'f', 2).unwrap();
        assert_eq!(12, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'h', 3).unwrap();
        assert_eq!(13, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'i', 3).unwrap();
        assert_eq!(14, threshold.value);
        assert!(!threshold.explored);
    }

    #[test]
    fn relaxed_computes_local_bounds_and_thresholds_with_pruning() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 15,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDFC::new();
        let result = mdd.compile(&input);
        assert!(result.is_ok());

        assert!(!mdd.is_exact());
        assert_eq!(Some(16), mdd.best_value());

        let mut v = FxHashMap::<char, isize>::default();
        mdd.drain_cutset(|n| {
            v.insert(*n.state, n.ub);
        });

        assert_eq!(16, v[&'a']);
        assert_eq!(14, v[&'b']);
        assert_eq!(2, v.len());

        assert!(cache.get_threshold(&'r', 0).is_some());
        assert!(cache.get_threshold(&'a', 1).is_some());
        assert!(cache.get_threshold(&'b', 1).is_some());
        assert!(cache.get_threshold(&'M', 2).is_none());
        assert!(cache.get_threshold(&'e', 2).is_some());
        assert!(cache.get_threshold(&'f', 2).is_some());
        assert!(cache.get_threshold(&'g', 3).is_none());
        assert!(cache.get_threshold(&'h', 3).is_some());
        assert!(cache.get_threshold(&'i', 3).is_some());
        assert!(cache.get_threshold(&'t', 4).is_none());

        let mut threshold = cache.get_threshold(&'r', 0).unwrap();
        assert_eq!(0, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'a', 1).unwrap();
        assert_eq!(10, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'b', 1).unwrap();
        assert_eq!(8, threshold.value);
        assert!(!threshold.explored);

        threshold = cache.get_threshold(&'e', 2).unwrap();
        assert_eq!(15, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'f', 2).unwrap();
        assert_eq!(13, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'h', 3).unwrap();
        assert_eq!(15, threshold.value);
        assert!(threshold.explored);

        threshold = cache.get_threshold(&'i', 3).unwrap();
        assert_eq!(15, threshold.value);
        assert!(threshold.explored);
    }

    #[test]
    fn test_default_visualisation() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 0,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDFC::new();
        let _ = mdd.compile(&input);

        let dot = include_str!("../../../../resources/visualisation_tests/default_viz.dot");
        let config = VizConfigBuilder::default().build().unwrap();
        let s = mdd.as_graphviz(&config);
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    #[test]
    fn test_terse_visualisation() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 0,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDFC::new();
        let _ = mdd.compile(&input);

        let dot = include_str!("../../../../resources/visualisation_tests/terse_viz.dot");
        let config = VizConfigBuilder::default()
            .show_value(false)
            .show_deleted(false)
            .show_rub(false)
            .show_locb(false)
            .show_threshold(false)
            .build()
            .unwrap();
        let s = mdd.as_graphviz(&config);
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    #[test]
    fn test_show_deleted_viz() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 0,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDFC::new();
        let _ = mdd.compile(&input);

        let dot = include_str!("../../../../resources/visualisation_tests/deleted_viz.dot");
        let config = VizConfigBuilder::default()
            .show_value(false)
            .show_deleted(true)
            .show_rub(false)
            .show_locb(false)
            .show_threshold(false)
            .build()
            .unwrap();
        let s = mdd.as_graphviz(&config);
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    #[test]
    fn test_show_group_merged() {
        let mut cache = SimpleCache::default();
        cache.initialize(&LocBoundsAndThresholdsExamplePb);
        let dominance = EmptyDominanceChecker::default();
        let input = CompilationInput {
            comp_type: crate::CompilationType::Relaxed,
            problem: &LocBoundsAndThresholdsExamplePb,
            relaxation: &LocBoundsAndThresholdsExampleRelax,
            ranking: &CmpChar,
            cutoff: &NoCutoff,
            max_width: 3,
            best_lb: 0,
            residual: &SubProblem {
                state: Arc::new('r'),
                value: 0,
                path: vec![],
                ub: isize::MAX,
                depth: 0,
            },
            cache: &cache,
            dominance: &dominance,
        };
        let mut mdd = DefaultMDDFC::new();
        let _ = mdd.compile(&input);

        let dot = include_str!("../../../../resources/visualisation_tests/clusters_viz.dot");
        let config = VizConfigBuilder::default()
            .show_value(false)
            .show_deleted(true)
            .show_rub(false)
            .show_locb(false)
            .show_threshold(false)
            .group_merged(true)
            .build()
            .unwrap();
        let s = mdd.as_graphviz(&config);
        assert_eq!(strip_format(dot), strip_format(&s));
    }

    fn strip_format(s: &str) -> String {
        s.lines().map(|l| l.trim()).collect()
    }

    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    struct DummyState {
        value: isize,
        depth: usize,
    }

    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    struct DummyDecisionState {
        value: isize,
    }

    #[derive(Copy, Clone)]
    struct DummyProblem;
    impl Problem for DummyProblem {
        type State = DummyState;
        type DecisionState = DummyDecisionState;

        fn nb_variables(&self) -> usize {
            3
        }
        fn initial_value(&self) -> isize {
            0
        }
        fn initial_state(&self) -> Self::State {
            DummyState { value: 0, depth: 0 }
        }

        fn transition(
            &self,
            state: &Self::State,
            decision: &crate::Decision<Self::DecisionState>,
        ) -> Self::State {
            DummyState {
                value: state.value + decision.value,
                depth: 1 + state.depth,
            }
        }

        fn transition_cost(
            &self,
            _: &Self::State,
            _: &Self::State,
            decision: &crate::Decision<Self::DecisionState>,
        ) -> isize {
            decision.value
        }

        fn next_variable(
            &self,
            depth: usize,
            _: &mut dyn Iterator<Item = &Self::State>,
        ) -> Option<crate::Variable> {
            if depth < self.nb_variables() {
                Some(Variable(depth))
            } else {
                None
            }
        }

        fn for_each_in_domain(
            &self,
            var: crate::Variable,
            _: &Self::State,
            f: &mut dyn DecisionCallback<Self::DecisionState>,
        ) {
            for d in 0..=2 {
                f.apply(Arc::new(Decision {
                    variable: var,
                    value: d,
                    state: None,
                }))
            }
        }
    }

    #[derive(Clone, Copy)]
    struct DummyInfeasibleProblem;
    impl Problem for DummyInfeasibleProblem {
        type State = DummyState;
        type DecisionState = DummyDecisionState;

        fn nb_variables(&self) -> usize {
            3
        }
        fn initial_value(&self) -> isize {
            0
        }
        fn initial_state(&self) -> Self::State {
            DummyState { value: 0, depth: 0 }
        }

        fn transition(
            &self,
            state: &Self::State,
            decision: &crate::Decision<Self::DecisionState>,
        ) -> Self::State {
            DummyState {
                value: state.value + decision.value,
                depth: 1 + state.depth,
            }
        }

        fn transition_cost(
            &self,
            _: &Self::State,
            _: &Self::State,
            decision: &crate::Decision<Self::DecisionState>,
        ) -> isize {
            decision.value
        }

        fn next_variable(
            &self,
            depth: usize,
            _: &mut dyn Iterator<Item = &Self::State>,
        ) -> Option<crate::Variable> {
            if depth < self.nb_variables() {
                Some(Variable(depth))
            } else {
                None
            }
        }

        fn for_each_in_domain(
            &self,
            _: crate::Variable,
            _: &Self::State,
            _: &mut dyn DecisionCallback<Self::DecisionState>,
        ) {
            /* do nothing, just consider that all domains are empty */
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRelax;
    impl Relaxation for DummyRelax {
        type State = DummyState;
        type DecisionState = DummyDecisionState;

        fn merge(&self, s: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
            s.next()
                .map(|s| DummyState {
                    value: 100,
                    depth: s.depth,
                })
                .unwrap()
        }
        fn relax(
            &self,
            _: &Self::State,
            _: &Self::State,
            _: &Self::State,
            _: &Decision<Self::DecisionState>,
            _: isize,
        ) -> isize {
            20
        }
        fn fast_upper_bound(&self, state: &Self::State) -> isize {
            (DummyProblem.nb_variables() - state.depth) as isize * 10
        }
    }

    #[derive(Copy, Clone)]
    struct DummyRanking;
    impl StateRanking for DummyRanking {
        type State = DummyState;
        type DecisionState = DummyDecisionState;

        fn compare(&self, a: &Self::State, b: &Self::State) -> Ordering {
            a.value.cmp(&b.value).reverse()
        }
    }
}
