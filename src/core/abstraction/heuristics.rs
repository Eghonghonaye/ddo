use crate::core::abstraction::mdd::{MDD, Node};
use std::hash::Hash;
use crate::core::abstraction::dp::{Variable, VarSet};

pub trait WidthHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn max_width(&self, dd: &dyn MDD<T, N>) -> usize;
}

pub trait VariableHeuristic<T, N>
    where T : Clone + Hash + Eq,
          N : Node<T, N> {
    fn next_var(&self, dd: &dyn MDD<T, N>, vars: &VarSet) -> Option<Variable>;
}