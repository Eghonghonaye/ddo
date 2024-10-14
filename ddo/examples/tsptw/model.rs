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
//! of the TSP+TW. (Implementation of the `Problem` trait).

use ddo::{Decision, Problem, StateRanking, Variable};
use smallbitset::Set256;

use crate::{instance::TsptwInstance, state::{ElapsedTime, Position, TsptwState}};

use crate::heuristics::TsptwRanking;
use clustering::kmeans;



/// This is the structure encapsulating the Tsptw problem.
#[derive(Clone)]
pub struct Tsptw {
    pub instance: TsptwInstance,
    pub initial : TsptwState,
    /// Whether we split edges by clustering,
    pub clustering: bool,
}
impl Tsptw {
    pub fn new(inst: TsptwInstance,clustering: bool) -> Self {
        let mut must_visit = Set256::default();
        (1..inst.nb_nodes).for_each(|i| {must_visit.add_inplace(i as usize);});
        let state = TsptwState {
            position  : Position::Node(0),
            elapsed   : ElapsedTime::FixedAmount{duration: 0},
            must_visit,
            maybe_visit: None,
            depth : 0
        };
        Self { instance: inst, initial: state, clustering }
    }
}
#[derive(Eq, PartialEq, Clone)]
pub struct StateClusterHelper {
    pub id: usize,
    pub cost: isize,
    pub state: TsptwState,
}

impl StateClusterHelper {
    fn new(id: usize, cost: isize, state: TsptwState) -> Self {
        StateClusterHelper { id, cost, state }
    }

    // fn from_capacity(depth: usize, capacity: usize) -> Self {
    //     StateClusterHelper {
    //         id: 0,
    //         state: KnapsackState { depth, capacity },
    //     }
    // }
}

impl clustering::Elem for StateClusterHelper {
    fn dimensions(&self) -> usize {
        1
    }

    fn at(&self, i: usize) -> f64 {
        // self.cost as f64
        match i {
            0 => match self.state.elapsed {
                ElapsedTime::FixedAmount{duration} => 
                    duration as f64,
                ElapsedTime::FuzzyAmount{earliest,latest:_} =>
                    earliest as f64
            },
            1 => self.cost as f64,
            _ => 0 as f64
        }
    }
}

impl Problem for Tsptw {
    type State = TsptwState;

    fn nb_variables(&self) -> usize {
        self.instance.nb_nodes as usize
    }

    fn initial_state(&self) -> TsptwState {
        self.initial.clone()
    }

    fn initial_value(&self) -> isize {
        0
    }
    
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        // When we are at the end of the tour, the only possible destination is
        // to go back to the depot. Any state that violates this constraint is
        // de facto infeasible.
        if state.depth as usize == self.nb_variables() - 1 {
            if self.can_move_to(state, 0) {
                f.apply(Decision { variable, value: 0 })
            }
            return;
        }

        for i in state.must_visit.iter() {
            if !self.can_move_to(state, i) {
                return;
            }
        }
        for i in state.must_visit.iter() {
            f.apply(Decision { variable, value: i as isize })
        }

        // Add those that can possibly be visited
        if let Some(maybe_visit) = &state.maybe_visit {
            for i in maybe_visit.iter() {
                if self.can_move_to(state, i) {
                    f.apply(Decision { variable, value: i as isize })
                }
            }
        }
    }

    fn transition(&self, state: &TsptwState, d: Decision) -> TsptwState {
        // if it is a true move
        let mut remaining = state.must_visit;
        remaining.remove_inplace(d.value as usize);
        // if it is a possible move
        let mut maybes = state.maybe_visit;
        if let Some(maybe) = maybes.as_mut() {
            maybe.remove_inplace(d.value as usize);
        }

        let time = self.arrival_time(state, d.value as usize);

        TsptwState {
            position : Position::Node(d.value as u16),
            elapsed  : time,
            must_visit: remaining,
            maybe_visit: maybes,
            depth: state.depth + 1
        }
    }

    fn transition_cost(&self, state: &TsptwState, _: &Self::State, d: Decision) -> isize {
        // Tsptw is a minimization problem but the solver works with a 
        // maximization perspective. So we have to negate the min if we want to
        // yield a lower bound.
        let twj = self.instance.timewindows[d.value as usize];
        let travel_time = self.min_distance_to(state, d.value as usize);
        let waiting_time = match state.elapsed {
            ElapsedTime::FixedAmount{duration} => 
                if (duration + travel_time) < twj.earliest {
                    twj.earliest - (duration + travel_time)
                } else {
                    0
                },
            ElapsedTime::FuzzyAmount{earliest, ..} => 
                if (earliest + travel_time) < twj.earliest {
                    twj.earliest - (earliest + travel_time)
                } else {
                    0
                }
        };

        -( (travel_time + waiting_time) as isize)
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<Variable> {
        if depth == self.nb_variables() {
            None
        } else {
            Some(Variable(depth))
        }
    }

    // False if feasible
    fn filter(&self, state: &Self::State, decision: &Decision) -> bool {
        if state.depth as usize == self.nb_variables() - 1 {
            if decision.value as usize > 0 {
                return true
            }
            else {
                return false
            }
    }

        if state.must_visit.contains(decision.value as usize) {
            // for i in state.must_visit.iter() {
            //     if !self.can_move_to(state, i) {
            //         return true;
            //     }
            // }
            // return false;
            return !self.can_move_to(state, decision.value as usize); 
        }
        else if let Some(maybe_visit) = &state.maybe_visit {
            if maybe_visit.contains(decision.value as usize) {
                return !self.can_move_to(state, decision.value as usize); 
            }
            else{
                return true;
            }
        }
        
        else{
            return true;
        }
    }

    fn split_edges(
        &self,
        decisions: &mut dyn Iterator<Item = (usize, isize, &Decision, &Self::State)>,
        how_many: usize,
    ) -> Vec<Vec<usize>> {
        if self.clustering {
            // println!("splitting {:?}", decisions.clone().map(|(a,b,_c)| (a,b)).collect::<Vec<_>>());
            let all_decision_state_capacities = decisions
                .map(|(id, cost, _d,s)| StateClusterHelper::new(id, cost, s.clone()))
                .collect::<Vec<_>>();
            let nclusters = usize::min(
                how_many,
                all_decision_state_capacities.len(),
            );
            let clustering = kmeans(nclusters, &all_decision_state_capacities, 100);
            let mut result = vec![Vec::new(); nclusters];
            for (label, h) in clustering.membership.into_iter().zip(clustering.elements) {
                result[label].push(h.id);
            }
            result.retain(|v| !v.is_empty()); 

            while result.len() < nclusters {
                
                result.sort_unstable_by(|a,b|a.len().cmp(&b.len()).reverse());
                let largest = result[0].clone();
                // println!("in while with {:?} of {:?} clusters and largest {:?}", result.len(),nclusters,largest);
                
                // remove largest from cluster
                result.remove(0);

                // extend what is left
                let diff = (nclusters - result.len()).min(largest.len());
                let mut split = vec![vec![]; diff];

                for (i, val) in largest.iter().copied().enumerate() {
                    split[i.min(diff-1)].push(val);
                }
                result.append(&mut split);
                
                // println!(
                //             "split into sizes: {:?}",
                //             result.iter().map(Vec::len).collect::<Vec<_>>()
                //         );

            }

            result
        } else {
            //TODO use split at mut logic of earlier, order, split at mut and them map into 2 vectors instead -- check keep merge art of code

            let mut all_decisions = decisions.collect::<Vec<_>>();
            //TODO confirm behaviour of ordering
            // all_decisions.sort_unstable_by(|(_a_id, a_cost, _a_dec,a_state), (_b_id, b_cost, _b_dec,b_state)| 
            //     {a_cost.cmp(b_cost)
            //     .then_with(|| TsptwRanking.compare(a_state, b_state)) 
            //     .reverse()}); //reverse because greater means more likely to be uniquely represented

            all_decisions.sort_unstable_by(|(_a_id, a_cost, _a_dec,a_state), (_b_id, b_cost, _b_dec,b_state)| 
                {a_cost.cmp(b_cost)
                .reverse()}); //reverse because greater means more likely to be uniquely represented

            let nclusters = usize::min(
                how_many,
                all_decisions.len(),
            );
            // reserve split vector lengths
            let mut split = vec![vec![]; nclusters];
            for (i, (d_id,_d_cost,_,_)) in all_decisions.iter().copied().enumerate() {
                split[i.min(nclusters-1)].push(d_id);
            }
            split
        }
    }
}

impl Tsptw {
    pub fn can_move_to(&self, state: &TsptwState, j: usize) -> bool {
        let twj         = self.instance.timewindows[j];
        let min_arrival = state.elapsed.add_duration(self.min_distance_to(state, j));
        match min_arrival {
            ElapsedTime::FixedAmount{duration}     => duration <= twj.latest,
            ElapsedTime::FuzzyAmount{earliest, ..} => earliest <= twj.latest,
        }
    }
    fn arrival_time(&self, state: &TsptwState, j: usize) -> ElapsedTime {
       let min_arrival = state.elapsed.add_duration(self.min_distance_to(state, j));
       let max_arrival = state.elapsed.add_duration(self.max_distance_to(state, j));

       let min_arrival = match min_arrival {
           ElapsedTime::FixedAmount{duration}     => duration,
           ElapsedTime::FuzzyAmount{earliest, ..} => earliest
       };
       let max_arrival = match max_arrival {
           ElapsedTime::FixedAmount{duration}    => duration,
           ElapsedTime::FuzzyAmount{latest, ..}  => latest
       };
       // This would be the arrival time if we never had to wait.
       let arrival_time = 
           if min_arrival.eq(&max_arrival) { 
               ElapsedTime::FixedAmount{duration: min_arrival} 
           } else {
               ElapsedTime::FuzzyAmount{earliest: min_arrival, latest: max_arrival}
           };
       // In order to account for the possible waiting time, we need to adjust
       // the earliest arrival time
       let twj = self.instance.timewindows[j];
       match arrival_time {
          ElapsedTime::FixedAmount{duration} => {
              ElapsedTime::FixedAmount{duration: duration.max(twj.earliest)}
          },
          ElapsedTime::FuzzyAmount{mut earliest, mut latest} => {
            earliest = earliest.max(twj.earliest);
            latest   = latest.min(twj.latest);

            if earliest.eq(&latest) {
                ElapsedTime::FixedAmount{duration: earliest}
            } else {
                ElapsedTime::FuzzyAmount{earliest, latest}
            }
          },
      }
    }
    fn min_distance_to(&self, state: &TsptwState, j: usize) -> usize {
        match &state.position {
            Position::Node(i) => self.instance.distances[*i as usize][j],
            Position::Virtual(candidates) => 
                candidates.iter()
                    .map(|i| self.instance.distances[i][j])
                    .min()
                    .unwrap()
        }
    }
    fn max_distance_to(&self, state: &TsptwState, j: usize) -> usize {
        match &state.position {
            Position::Node(i) => self.instance.distances[*i as usize][j],
            Position::Virtual(candidates) => 
                candidates.iter()
                    .map(|i| self.instance.distances[i][j])
                    .max()
                    .unwrap()
        }
    }
}
