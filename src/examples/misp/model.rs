use crate::examples::misp::instance::Graph;
use crate::core::abstraction::dp::{Variable, Problem, Decision, VarSet};
use bitset_fixed::BitSet;
use std::ops::Not;
use std::fs::File;
use std::io::{Read, BufReader, BufRead, Lines};

pub struct Misp {
    pub graph : Graph,
    yes_no    : Vec<i32>,
    no        : Vec<i32>
}

impl Misp {
    pub fn new(mut graph : Graph) -> Misp {
        graph.complement();
        Misp {graph, yes_no: vec![1, 0], no: vec![0]}
    }
}

impl Problem<BitSet> for Misp {
    fn nb_vars(&self) -> usize {
        self.graph.nb_vars
    }

    fn initial_state(&self) -> BitSet {
        BitSet::new(self.graph.nb_vars).not()
    }

    fn initial_value(&self) -> i32 {
        0
    }

    fn domain_of(&self, state: &BitSet, var: Variable) -> &[i32] {
        if state[var.0] { &self.yes_no } else { &self.no }
    }

    fn transition(&self, state: &BitSet, _vars: &VarSet, d: Decision) -> BitSet {
        let mut bs = state.clone();
        bs.set(d.variable.0, false);

        // drop adjacent vertices if needed
        if d.value == 1 {
            bs &= &self.graph.adj_matrix[d.variable.0];
        }

        bs
    }

    fn transition_cost(&self, _state: &BitSet, _vars: &VarSet, d: Decision) -> i32 {
        if d.value == 0 {
            0
        } else {
            self.graph.weights[d.variable.0]
        }
    }

    fn impacted_by(&self, state: &BitSet, variable: Variable) -> bool {
        state[variable.0]
    }
}
impl From<File> for Misp {
    fn from(file: File) -> Self {
        BufReader::new(file).into()
    }
}
impl <S: Read> From<BufReader<S>> for Misp {
    fn from(buf: BufReader<S>) -> Self {
        buf.lines().into()
    }
}
impl <B: BufRead> From<Lines<B>> for Misp {
    fn from(lines: Lines<B>) -> Self {
        Self::new(lines.into())
    }
}