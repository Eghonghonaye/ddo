use std::cmp::min;

use crate::core::abstraction::dp::{Problem, Relaxation};
use crate::core::abstraction::mdd::{MDD, Node, Arc};
use crate::examples::max2sat::model::{Max2Sat, State};
use std::rc::Rc;

#[derive(Debug)]
pub struct Max2SatRelax<'a> {
    problem : &'a Max2Sat
}

impl <'a> Max2SatRelax<'a> {
    pub fn new(problem: &'a Max2Sat) -> Max2SatRelax<'a> {
        Max2SatRelax { problem }
    }
}

impl Relaxation<State> for Max2SatRelax<'_> {
    fn merge_nodes(&self, _dd: &dyn MDD<State>, nodes: &[&Node<State>]) -> Node<State> {
        let mut benefits      = vec![0; self.problem.nb_vars()];
        let mut relaxed_costs = nodes.iter().cloned().map(|n| n.lp_len).collect::<Vec<i32>>();

        // Compute the merged state and relax the best edges costs
        let vars = self.problem.all_vars();
        for v in vars.iter() {
            let mut sign      = 0;
            let mut min_benef = i32::max_value();
            let mut same      = true;

            for node in nodes.iter().cloned() {
                let substate = node.state[v];
                min_benef = min(min_benef, substate.abs());

                if sign == 0 && substate != 0 {
                    sign = substate.abs() / substate;
                } else if sign * substate < 0 {
                    same = false;
                    break;
                }
            }

            if same {
                benefits[v.0] = sign * min_benef;
            }

            for j in 0..nodes.len() {
                relaxed_costs[j] += nodes[j].state[v].abs() - benefits[v.0].abs();
            }
        }

        // Find the best incoming edge
        let mut lp_len    = i32::min_value();
        let mut input_arc = None;

        for (j, node) in nodes.iter().cloned().enumerate() {
            if relaxed_costs[j] > lp_len {
                lp_len    = relaxed_costs[j];

                let nd_arc= node.lp_arc.as_ref().unwrap();
                input_arc = Some(Arc {src: Rc::clone(&nd_arc.src), decision: nd_arc.decision, weight: lp_len - nd_arc.src.lp_len});
            }
        }

        Node::new(State(benefits), lp_len, input_arc, false)
    }
}