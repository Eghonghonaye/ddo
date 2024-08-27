// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This module contains the definition of the dynamic programming formulation 
//! of the SOP. (Implementation of the `Problem` trait).

use ddo::{Problem, Variable, Decision, DecisionCallback};
use std::{ops::Range, sync::Arc};

use crate::{io_utils::SopInstance, state::{Previous, SopDecisionState, SopState}, BitSet};
use clustering::kmeans;

/// This is the structure encapsulating the Sop problem.
#[derive(Debug, Clone)]
pub struct Sop {
    pub instance: SopInstance,
    pub initial : SopState,
    pub cheapest_edges: Vec<Vec<(isize, usize)>>,
    /// Whether we split edges by clustering,
    clustering: u8,
}
impl Sop {
    pub fn new(inst: SopInstance, clustering: u8) -> Self {
        let cheapest_edges: Vec<Vec<(isize, usize)>> = Self::compute_cheapest_edges(&inst);
        let mut must_schedule = BitSet::default();
        (1..inst.nb_jobs).for_each(|i| {must_schedule.add_inplace(i as usize);});
        let state = SopState {
            previous: Previous::Job(0),
            must_schedule,
            maybe_schedule: None,
            depth : 0
        };
        Self { instance: inst, initial: state, cheapest_edges, clustering }
    }

    fn compute_cheapest_edges(inst: &SopInstance) -> Vec<Vec<(isize, usize)>> {
        let mut cheapest = vec![];
        let n = inst.nb_jobs as usize;
        for i in 0..n {
            let mut cheapest_to_i = vec![];
            for j in 0..n {
                if i == j || inst.distances[j][i] == -1 {
                    continue;
                }
                cheapest_to_i.push((inst.distances[j][i], j));
            }
            cheapest_to_i.sort_unstable();
            cheapest.push(cheapest_to_i);
        }
        cheapest
    }
    //helper function to create state edges, basically does the same as transition
    pub fn update_decision_state(&self,state: &SopState,decision: &mut Decision<SopDecisionState>){
        let next = self.transition(state, decision);
        decision.state =  Some(SopDecisionState{
            previous: next.previous,
            must_schedule: next.must_schedule,
            maybe_schedule: next.maybe_schedule
        });
    }
    
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub struct StateClusterHelper {
    pub id: usize,
    pub cost: isize,
}
impl StateClusterHelper {
    fn new(id: usize, cost: isize) -> Self {
        StateClusterHelper { id, cost }
    }
}
impl clustering::Elem for StateClusterHelper {
    fn dimensions(&self) -> usize {
        1
    }

    fn at(&self, _i: usize) -> f64 {
        self.cost as f64
    }
}

impl Problem for Sop {
    type State = SopState;
    type DecisionState = SopDecisionState;

    fn nb_variables(&self) -> usize {
        // -1 because we always start from the first job
        (self.instance.nb_jobs - 1) as usize
    }

    fn initial_state(&self) -> SopState {
        self.initial.clone()
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback<Self::DecisionState>) {
        // When we are at the end of the schedule, the only possible destination is
        // to go to the last job.
        if state.depth as usize == self.nb_variables() - 1 {
            let mut decision = Decision { variable, value: (self.instance.nb_jobs - 1) as isize, state: None};
            self.update_decision_state(state,&mut decision);
            f.apply(Arc::new(decision));
        } else {
            for i in state.must_schedule.iter() {
                if self.can_schedule(state, i) {
                    let mut decision = Decision { variable, value: i as isize, state: None};
                    self.update_decision_state(state,&mut decision);
                    f.apply(Arc::new(decision))
                }
            }
    
            // Add those that can possibly be scheduled
            if let Some(maybe_visit) = &state.maybe_schedule {
                for i in maybe_visit.iter() {
                    if self.can_schedule(state, i) {
                        let mut decision = Decision { variable, value: i as isize, state: None};
                        self.update_decision_state(state,&mut decision);
                        f.apply(Arc::new(decision))
                    }
                }
            }
        }
    }

    fn transition(&self, state: &SopState, d: &Decision<SopDecisionState>) -> SopState {
        let job = d.value as usize;

        let mut next = *state;

        next.previous = Previous::Job(job);
        next.depth += 1;
        
        next.must_schedule.remove_inplace(job);
        if let Some(maybe) = next.maybe_schedule.as_mut() {
            maybe.remove_inplace(job);
        }
        next
    }

    fn transition_cost(&self, state: &SopState, _: &Self::State, d: &Decision<SopDecisionState>) -> isize {
        // Sop is a minimization problem but the solver works with a 
        // maximization perspective. So we have to negate the cost.

        - self.min_distance_to(state, d.value as usize)
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable> {
        if depth < self.nb_variables() {
            Some(Variable(depth))
        } else {
            None
        }
    }

    fn filter(&self, state: &Self::State, decision: &Decision<Self::DecisionState>) -> bool {
        // what is infeasible?
        if state.depth as usize == self.nb_variables() - 1 {
            return decision.value != (self.instance.nb_jobs - 1) as isize;
        } else {
            return !self.can_schedule(state,decision.value as usize);
        }
    }

    fn split_state_edges(
        &self,
        _state: &Self::State,
        decisions: &mut dyn Iterator<Item = (usize, isize, &Decision<Self::DecisionState>)>,
    ) -> Vec<Vec<usize>> {

        if self.clustering >= 2 {
            let all_decision_costs = decisions
                .map(|(id,cost, _d)| StateClusterHelper::new(id, cost))
                .collect::<Vec<_>>();
            let nclusters = usize::min(
                self.clustering as usize,
                all_decision_costs.len(),
            );
            let clustering = kmeans(nclusters, &all_decision_costs, 100);
            let mut result = vec![Vec::new(); nclusters];
            for (label, h) in clustering.membership.into_iter().zip(clustering.elements) {
                result[label].push(h.id);
            }
            result.retain(|v| !v.is_empty());

            if result.len() == 1 {
                let split = result[0].split_at(result[0].len() - 1);
                result = vec![split.0.to_vec(), split.1.to_vec()];
            } else {
                println!(
                    "split into sizes: {:?}",
                    result.iter().map(Vec::len).collect::<Vec<_>>()
                );
            }

            result
        } else {
            //TODO use split at mut logic of earlier, order, split at mut and them map into 2 vectors instead -- check keep merge art of code

            let mut all_decisions = decisions.collect::<Vec<_>>();
            //TODO confirm behaviour of ordering
            all_decisions.sort_unstable_by(|(_a_id, a_cost, _a_d), (_b_id, b_cost, _b_d)| 
               b_cost.cmp(&a_cost)); //TODO: am i prioritising best or worst
            let split_point = all_decisions.len() - 1;
            let (a, b) = all_decisions.split_at_mut(split_point);

            let split_a = a.iter().map(|(x, _,_)| *x).collect();
            let split_b = b.iter().map(|(x, _,_)| *x).collect();

            vec![split_a, split_b]
        }

    }

    fn check_conflict(&self, in_decision: &Decision<Self::DecisionState>, out_decision: &Decision<Self::DecisionState>) -> bool {
        return self.conflicting_decisions(in_decision,out_decision)
    }
    fn value_range(&self) -> Range<isize> {
       0..self.nb_variables() as isize
    }
}

impl Sop {
    pub fn can_schedule(&self, state: &SopState, j: usize) -> bool {
        let maybe_scheduled = match &state.maybe_schedule {
            Some(maybes) => (state.must_schedule.union(*maybes)).flip(),
            None => state.must_schedule.flip(),
        };
        maybe_scheduled.contains_all(self.instance.predecessors[j])
    }
    pub fn min_distance_to(&self, state: &SopState, j: usize) -> isize {
        match &state.previous {
            Previous::Job(i) => if self.instance.distances[*i as usize][j] == -1 {
                isize::MAX
            } else {
                self.instance.distances[*i as usize][j]
            },
            Previous::Virtual(candidates) => 
                candidates.iter()
                    .map(|i| self.instance.distances[i as usize][j as usize])
                    .filter(|w| *w != -1)
                    .min()
                    .unwrap()
        }
    }
    pub fn conflicting_decisions(&self, in_decision: &Decision<SopDecisionState>,out_decision: &Decision<SopDecisionState>) -> bool{
        if let Some(state) = in_decision.state {
            let maybe_scheduled = match &state.maybe_schedule {
                Some(maybes) => (state.must_schedule.union(*maybes)).flip(),
                None => state.must_schedule.flip(),
            };
            maybe_scheduled.contains_all(self.instance.predecessors[out_decision.value as usize])}
        else{
            false
        }
    }
}
