// use crate::abstraction::state::{State, DecisionState};
// use ddo::{Problem, Variable, Decision};
use ddo::*;
use crate::abstraction::instance::{Instance,OpId};
use crate::abstraction::constraints::{SetupType,Constraint};
// use crate::implementation::constraints::release;
// use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::cmp;

// extern crate bitvector;
use crate::utils::bitvector::*;

#[derive(Clone, Hash)]
pub struct State{
    pub last_decision: Vec<Rc<Decision<DecisionState>>>,
    pub depth: u16,
    pub est: Vec<usize>, // indexed by opid
    pub lst: Vec<usize>, // indexed by opid
    pub availability: Vec<usize>, // indexed by mid
    pub def_scheduled: BitVector,
    pub maybe_scheduled: BitVector,
    pub feasible_set: BitVector,   
}
impl Default for State{
    fn default() -> Self {
        Self {
            last_decision: vec![],
            depth : 0,
            est: vec![],
            lst: vec![],
            availability : vec![],
            def_scheduled : BitVector::new(0),
            maybe_scheduled : BitVector::new(0),
            feasible_set : BitVector::new(0), // all operations???
            }
    }
}
impl PartialEq for State{
    // Required method
    fn eq(&self, other: &Self) -> bool{
        
        if self.depth == other.depth {} else {return false;};
        if self.est == other.est {} else {return false;};
        if self.lst == other.lst {} else {return false;};
        if self.availability == other.availability {} else {return false;};

        // TODO: this notion of equality seems too strict - revise
        // TODO: bitvector implementation confirm
        if other.def_scheduled.iter().all(|item| self.def_scheduled.contains(item)) {} else {return false;};
        if other.maybe_scheduled.iter().all(|item| self.maybe_scheduled.contains(item)) {} else {return false;};
        if other.feasible_set.iter().all(|item| self.feasible_set.contains(item)) {} else {return false;};

        true
    }

    // Provided method
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
     }
}

impl Eq for State {}

#[derive(Debug, Clone, Eq, Hash)]
pub struct DecisionState{
    est: Vec<u8>,
    lst: Vec<u8>,
    avail: Vec<u8>
}
impl PartialEq for DecisionState{
    // Required method
    fn eq(&self, _other: &Self) -> bool{
        true
    }

    // Provided method
    fn ne(&self, _other: &Self) -> bool {
        true
     }
}

pub struct Mms{
    pub instance: Instance,
    pub initial_state: State
}

impl Mms{
    pub fn new() -> Self{
        Self{instance: Instance::default(),initial_state: State::default()}
    }
    pub fn initialise(problem_instance: Instance) -> Self{
        let state = State {
            last_decision: vec![],
            depth : 0,
            est: vec![usize::MIN;problem_instance.nops],
            lst: vec![usize::MAX;problem_instance.nops],
            def_scheduled : BitVector::new(problem_instance.nops),
            maybe_scheduled : BitVector::new(problem_instance.nops),
            feasible_set : BitVector::new(problem_instance.nops), // all operations???
            availability : vec![usize::MIN;problem_instance.nmachs]
        };
        Self{instance: problem_instance, initial_state: state}
    }
}

pub struct MmsRelax<'a>{
    pub problem: &'a Mms,
}

pub struct MmsDominance;

impl Problem for Mms{
    /// The DP model of the problem manipulates a state which is user-defined.
    /// Any type implementing Problem must thus specify the type of its state.
    type State = State;
    type DecisionState = DecisionState;
    /// Any problem bears on a number of variable $x_0, x_1, x_2, ... , x_{n-1}$
    /// This method returns the value of the number $n$
    fn nb_variables(&self) -> usize{
        self.instance.ops.len()
    }
    /// This method returns the initial state of the problem (the state of $r$).
    fn initial_state(&self) -> Self::State{
        self.initial_state.clone()
    }
    /// This method returns the initial value $v_r$ of the problem
    fn initial_value(&self) -> isize{
        0
    }
    /// This method is an implementation of the transition function mentioned
    /// in the mathematical model of a DP formulation for some problem.
    fn transition(&self, state: &Self::State, decision: &Decision<DecisionState>) -> Self::State{
        // update all the values according to the paper rules
        let op = OpId::new(decision.value.try_into().unwrap());
        let mut def_sched = state.def_scheduled.clone();
        def_sched.insert(decision.value.try_into().unwrap()); // should be some opid but that its in the decision does not mean it was defintely scheduled
        let mut may_sched = state.maybe_scheduled.clone();
        may_sched.insert(decision.value.try_into().unwrap());

        let feasible_set = self.instance.construct_feasible_set(state); // initialise feasible set as all operations in the system
        let mut pure_seq: usize = 0;
        let mut machine_seq: usize = 0;
        let mut indpndt_ops_seq: usize = 0;
        let mut indpndt_mch_seq: usize = 0;

        for constraint in &self.instance.constraints[&op]{
            match constraint {
                Constraint::SetupCons(cons) => {
                    match cons.setup_type{
                        SetupType::PureSequence => pure_seq = {
                                                            for y in &state.last_decision {
                                                                if OpId::new(y.value.try_into().unwrap()) == cons.op_a.id {
                                                                    pure_seq = cmp::min(pure_seq,cons.value);
                                                                    } 
                                                                }
                                                                pure_seq
                                                            },
                        SetupType::MachineSequence => machine_seq = if state.maybe_scheduled.contains(cons.op_a.id.as_usize()){cmp::min(machine_seq,cons.value)} else {machine_seq},
                        SetupType::IndependentOps => indpndt_ops_seq =  if state.def_scheduled.contains(cons.op_a.id.as_usize()){cmp::max(indpndt_ops_seq,cons.value)} else {indpndt_ops_seq},
                        SetupType::IndependentMachines => indpndt_mch_seq = cmp::max(indpndt_mch_seq,cons.value)}},
                _ => {}
            }
        }

        let setup = cmp::max(cmp::max(cmp::max(pure_seq,machine_seq),indpndt_ops_seq),indpndt_mch_seq);
        let release = self.instance.ops[&op].release;
        let processing = self.instance.ops[&op].processing;
        let machine = self.instance.ops[&op].machine;
        // must update est, lst and availability in this order
        let mut est = state.est.clone();
        est[op.as_usize()] = cmp::max(release, state.availability[machine.as_usize()] + setup); // modifies est in place
        let lst = state.lst.clone();
        let mut availability = state.availability.clone();
        availability[machine.as_usize()] = if est[op.as_usize()] == usize::MIN {processing} else {est[op.as_usize()]+processing};
        let mut last_decision = state.last_decision.clone();
        last_decision.push(Rc::new(decision.clone()));

        let new_state = State{
            last_decision: last_decision.clone(),
            depth : state.depth + 1,
            est : est,
            lst : lst, // ignore deadlines for now
            def_scheduled : def_sched,
            maybe_scheduled : may_sched,
            feasible_set : feasible_set,
            availability : availability
        };

        new_state
    }
    /// This method is an implementation of the transition cost function mentioned
    /// in the mathematical model of a DP formulation for some problem.
    fn transition_cost(&self, source: &Self::State, dest: &Self::State, _decision: &Decision<DecisionState>) -> isize{
        let dest_obj = match dest.availability.iter().max_by(|a, b| a.cmp(&b)) {
            Some(x) => x,
            None => &0
        }; 
        
        let src_obj = match source.availability.iter().max_by(|a, b| a.cmp(&b)) {
                Some(x) => x,
                None => &0
            };

        (dest_obj - src_obj) as isize
       }
    /// Any problem needs to be able to specify an ordering on the variables
    /// in order to decide which variable should be assigned next. This choice
    /// is an **heuristic** choice. The variable ordering does not need to be
    /// fixed either. It may simply depend on the depth of the next layer or
    /// on the nodes that constitute it. These nodes are made accessible to this
    /// method as an iterator.
    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable>{
            // variables processed in fixed order, they represent positions in the schedule
            // no more variables = terminal
            if depth < self.nb_variables() {
                Some(Variable(depth))
            } else {
                None
            }
        }
    /// This method calls the function `f` for any value in the domain of 
    /// variable `var` when in state `state`.  The function `f` is a function
    /// (callback, closure, ..) that accepts one decision.
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback<Self::DecisionState>){
        for (_,op) in &self.instance.ops{
            if state.feasible_set.contains(op.id.as_usize()){
                f.apply(Arc::new(Decision{variable, value: op.id.as_isize(), state: None}));
            }
        }
        

    }
    /// This method returns false iff this node can be moved forward to the next
    /// layer without making any decision about the variable `_var`.
    /// When that is the case, a default decision is to be assumed about the 
    /// variable. Implementing this method is only ever useful if you intend to 
    /// compile a decision diagram that comprises long arcs.
    fn is_impacted_by(&self, _var: Variable, _state: &Self::State) -> bool {
        true
    }
}


pub struct MmsRanking;
impl StateRanking for MmsRanking {
    type State = State;
    type DecisionState = DecisionState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {

        let a_obj = match a.availability.iter().max_by(|a, b| a.cmp(&b)) {
            Some(x) => x,
            None => &0
        }; 
        
        let b_obj = match b.availability.iter().max_by(|a, b| a.cmp(&b)) {
                Some(x) => x,
                None => &0
            };
        a_obj.cmp(b_obj)
    }
}


impl<'a> Relaxation for MmsRelax<'a>{
    /// Similar to the DP model of the problem it relaxes, a relaxation operates
    /// on a set of states (the same as the problem). 
    type State = State;
    type DecisionState = DecisionState;

    /// This method implements the merge operation: it combines several `states`
    /// and yields a new state which is supposed to stand for all the other
    /// merged states. In the mathematical model, this operation was denoted
    /// with the $\oplus$ operator.
    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State{
        
        // update all the values according to the paper rules
        let mut last_decision: Vec<Rc<Decision<DecisionState>>> = vec![];
        let mut est_min: Vec<_> = vec![usize::MIN;self.problem.instance.nops];
        let mut lst_max: Vec<_> = vec![usize::MIN;self.problem.instance.nops];
        let mut availability_min: Vec<_> = vec![usize::MIN;self.problem.instance.nmachs];
        let mut def_sched_intersect = BitVector::ones(self.problem.instance.nops);
        let mut maybe_sched_union = BitVector::new(self.problem.instance.nops);
        let mut feasible_set_union = BitVector::new(self.problem.instance.nops);
        let mut depth = 0;

        for state in states{
            last_decision.extend(state.last_decision.clone());
            depth = cmp::min(depth,state.depth);


            est_min = est_min.iter().zip(state.est.iter()).map(|(&b, &v)| cmp::min(b,v)).collect::<Vec<_>>();
            lst_max = lst_max.iter().zip(state.lst.iter()).map(|(&b, &v)| cmp::max(b,v)).collect::<Vec<_>>();
            availability_min = availability_min.iter().zip(state.availability.iter()).map(|(&b, &v)| cmp::min(b,v)).collect::<Vec<_>>();


            def_sched_intersect.intersection_inplace(&state.def_scheduled);
            maybe_sched_union.union_inplace(&state.maybe_scheduled);
            feasible_set_union.union_inplace(&state.feasible_set);
        }        

        // mostly getting minimums
        let new_state = State{
            last_decision: last_decision,
            depth : depth,
            est : est_min,
            lst : lst_max, // ignore deadlines for now
            def_scheduled : def_sched_intersect,
            maybe_scheduled : maybe_sched_union,
            feasible_set : feasible_set_union,
            availability : availability_min
        };

        
        new_state

    }
    
    /// This method relaxes the cost associated to a particular decision. It
    /// is called for any arc labeled `decision` whose weight needs to be 
    /// adjusted because it is redirected from connecting `src` with `dst` to 
    /// connecting `src` with `new`. In the mathematical model, this operation
    /// is denoted by the operator $\Gamma$.
    fn relax(
        &self,
        source: &Self::State,
        _dest: &Self::State,
        new: &Self::State,
        _decision: &Decision<Self::DecisionState>,
        _cost: isize,
    ) -> isize{
        // TODO: now that its just a vector and not a map, do we still need max_by? we likely should use a simpler/cheaper max
        let new_obj = match new.availability.iter().max_by(|a, b| a.cmp(&b)).map(|v| v) {
            Some(x) => x,
            None => &0
        }; 
        
        let src_obj = match source.availability.iter().max_by(|a, b| a.cmp(&b)).map(|v| v) {
                Some(x) => x,
                None => &0
            };

        (new_obj - src_obj) as isize
    }

    /// Returns a very rough estimation (upper bound) of the optimal value that 
    /// could be reached if state were the initial state
    fn fast_upper_bound(&self, _state: &Self::State) -> isize {
        isize::MAX
    }
}

/*
impl Dominance for MmsDominance{
    type State = MmsState;
    type Key = ;

    /// Takes a state and returns a key that maps it to comparable states
    fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key>{

    }

    /// Returns the number of dimensions to include in the comparison
    fn nb_dimensions(&self, state: &Self::State) -> usize{

    }

    /// Returns the i-th coordinate associated with the given state
    /// Greater is better for the dominance check
    fn get_coordinate(&self, state: &Self::State, i: usize) -> isize{

    }

    /// Whether to include the value as a coordinate in the dominance check
    fn use_value(&self) -> bool { 
        false 
    }

    /// Checks whether there is a dominance relation between the two states, given the coordinates
    /// provided by the function get_coordinate evaluated for all i in 0..self.nb_dimensions()
    /// Note: the states are assumed to have the same key, otherwise they are not comparable for dominance
    fn partial_cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Option<DominanceCmpResult> {
        let mut ordering = Ordering::Equal;
        for i in 0..self.nb_dimensions(a) {
            match (ordering, self.get_coordinate(a, i).cmp(&self.get_coordinate(b, i))) {
                (Ordering::Less, Ordering::Greater)  => return None,
                (Ordering::Greater, Ordering::Less)  => return None,
                (Ordering::Equal, Ordering::Greater) => ordering = Ordering::Greater,
                (Ordering::Equal, Ordering::Less)    => ordering = Ordering::Less,
                (_, _)                               => (),
            }
        }
        if self.use_value() {
            match (ordering, val_a.cmp(&val_b)) {
                (Ordering::Less, Ordering::Greater)  => None,
                (Ordering::Greater, Ordering::Less)  => None,
                (Ordering::Equal, Ordering::Greater) => Some(DominanceCmpResult { ordering: Ordering::Greater, only_val_diff: true }),
                (Ordering::Equal, Ordering::Less)    => Some(DominanceCmpResult { ordering: Ordering::Less, only_val_diff: true }),
                (_, _)                               => Some(DominanceCmpResult { ordering, only_val_diff: false }),
            }
        } else {
            Some(DominanceCmpResult { ordering, only_val_diff: false })
        }
    }

    /// Comparator to order states by increasing value, regardless of their key
    fn cmp(&self, a: &Self::State, val_a: isize, b: &Self::State, val_b: isize) -> Ordering {
        if self.use_value() {
            match val_a.cmp(&val_b) {
                Ordering::Less    => return Ordering::Less,
                Ordering::Greater => return Ordering::Greater,
                Ordering::Equal   => (),
            }
        }
        for i in 0..self.nb_dimensions(a) {
            match self.get_coordinate(a, i).cmp(&self.get_coordinate(b, i)) {
                Ordering::Less    => return Ordering::Less,
                Ordering::Greater => return Ordering::Greater,
                Ordering::Equal   => (),
            }
        }
        Ordering::Equal
    }

}
*/