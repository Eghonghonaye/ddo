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

//! This example show how to implement a solver for the knapsack problem using ddo.
//! It is a fairly simple example but  features most of the aspects you will want to
//! copy when implementing your own solver.
use std::{fs::{self, File}, io::{BufRead, BufReader}, num::ParseIntError, path::Path, sync::Arc, time::{Duration, Instant}};

use clap::Parser;
use clustering::kmeans;
use ddo::*;
use ordered_float::OrderedFloat;
use serde_json::json;
use tensorflow::Tensor;

#[cfg(test)]
mod tests;

/// In our DP model, we consider a state that simply consists of the remaining 
/// capacity in the knapsack. Additionally, we also consider the *depth* (number
/// of assigned variables) as part of the state since it useful when it comes to
/// determine the next variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KnapsackState {
    /// the number of variables that have already been decided upon in the complete
    /// problem.
    depth: usize,
    /// the remaining capacity in the knapsack. That is the maximum load the sack
    /// can bear without cracking **given what is already in the sack**.
    capacity: usize
}

/// This structure represents a particular instance of the knapsack problem.
/// This is the structure that will implement the knapsack model.
/// 
/// The problem definition is quite easy to understand: there is a knapsack having 
/// a maximum (weight) capacity, and a set of items to chose from. Each of these
/// items having a weight and a profit, the goal is to select the best subset of
/// the items to place them in the sack so as to maximize the profit.
pub struct Knapsack {
    /// The maximum capacity of the sack (when empty)
    capacity: usize,
    /// the profit of each item
    profit: Vec<isize>,
    /// the weight of each item.
    weight: Vec<usize>,
    /// the order in which the items are considered
    order: Vec<usize>,
    /// Whether we split edges by clustering,
    clustering: bool,
    // Optional ml model to support decision making
    ml_model: Option<TfModel>
}

impl Knapsack {
    pub fn new(capacity: usize, profit: Vec<isize>, weight: Vec<usize>, clustering: bool, ml_model:Option<TfModel>) -> Self {
        let mut order = (0..profit.len()).collect::<Vec<usize>>();
        order.sort_unstable_by_key(|i| OrderedFloat(- profit[*i] as f64 / weight[*i] as f64));

        Knapsack { capacity, profit, weight, order, clustering, ml_model }
    }
}

impl ModelHelper for Knapsack{
    type State = KnapsackState;
    //TODO - Calculate selection status properly
    fn state_to_input_tensor(&self,state: &Self::State) -> Result<Tensor<f32>, tensorflow::Status>{
        let values = self.profit.clone();
        let weights = self.weight.clone();
        let capacity = [state.capacity as isize];
        let selection_status = [1,1,1,1,1,0,0,0,0,0];
        let input: Vec<f32> = values.iter().map(|x| *x as f32).
                                chain(weights.iter().map(|x| *x as f32)).
                                chain(capacity.iter().map(|x| *x as f32)).
                                chain(selection_status.iter().map(|x| *x as f32)).
                                collect();

        let tensor = Tensor::new(&[1,(3*values.len()+1) as u64]).with_values(&input)?;
        println!("Output tensor is {:?}", tensor);
        Ok(tensor)
    }

    fn extract_decision_from_model_output(&self, state: &Self::State, output: Tensor<f32>) -> Option<Decision>{
        //TODO - Am I interpreting model output correctly?
        if let Some(variable) = self.next_variable(state.depth, &mut vec![].iter()){
            return Some(Decision {
                variable,
                value: if output[variable.0] > 0.5{TAKE_IT} else {LEAVE_IT_OUT},
                })
        }
        None
    }
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub struct StateClusterHelper {
    pub id: usize,
    pub state: KnapsackState,
}

impl StateClusterHelper {
    fn new(id: usize, state: KnapsackState) -> Self {
        StateClusterHelper { id, state }
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

    fn at(&self, _i: usize) -> f64 {
        self.state.capacity as f64
    }
}

/// For each variable in the decision problem, there are two possible choices:
/// either we take the item in the sack, or we decide to leave it out. This
/// constant is used to indicate that the item is to be taken in the sack.
const TAKE_IT: isize = 1;
/// For each variable in the decision problem, there are two possible choices:
/// either we take the item in the sack, or we decide to leave it out. This
/// constant is used to indicate that the item is to be left out of the sack.
const LEAVE_IT_OUT: isize = 0;

/// This is how you implement the labeled transition system (LTS) semantics of
/// a simple dynamic program solving the knapsack problem. The definition of
/// each of the methods should be pretty clear and easy to grasp. Should you
/// want more details on the role of each of these methods, then you are 
/// encouraged to go checking the documentation of the `Problem` trait.
impl Problem for Knapsack {
    type State = KnapsackState;

    fn nb_variables(&self) -> usize {
        self.profit.len()
    }
    fn for_each_in_domain(&self, variable: Variable, state: &Self::State, f: &mut dyn DecisionCallback)
    {
        if state.capacity >= self.weight[variable.id()] {
            f.apply(Decision { variable, value: TAKE_IT });
        }
        f.apply(Decision { variable, value: LEAVE_IT_OUT });
    }
    fn initial_state(&self) -> Self::State {
        KnapsackState{ depth: 0, capacity: self.capacity }
    }
    fn initial_value(&self) -> isize {
        0
    }
    fn transition(&self, state: &Self::State, dec: Decision) -> Self::State {
        let mut ret = *state;
        ret.depth  += 1;
        if dec.value == TAKE_IT { 
            ret.capacity -= self.weight[dec.variable.id()] 
        }
        ret
    }
    fn transition_cost(&self, _state: &Self::State, _: &Self::State, dec: Decision) -> isize {
        self.profit[dec.variable.id()] * dec.value
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<Variable> {
        let n = self.nb_variables();
        if depth < n {
            Some(Variable(self.order[depth]))
        } else {
            None
        }
    }

    fn filter(&self, state: &Self::State, decision: &Decision) -> bool {
        if decision.value == TAKE_IT{
            self.weight[decision.variable.id()] > state.capacity
        }
        else{
            false //we're always allowed to leave an item
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
                .map(|(id, _cost, _d,s)| StateClusterHelper::new(id, *s))
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
            all_decisions.sort_unstable_by(|(_a_id, a_cost, _a_dec,a_state), (_b_id, b_cost, _b_dec,b_state)| 
                {a_cost.cmp(b_cost)
                .then_with(|| KPRanking.compare(a_state, b_state)) 
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
            // let split_point = all_decisions.len() - 1;
            // let (a, b) = all_decisions.split_at_mut(split_point);

            // let split_a = a.iter().map(|(x, _,_)| *x).collect();
            // let split_b = b.iter().map(|(x, _,_)| *x).collect();

            // vec![split_a, split_b]
        }
    }

    fn perform_ml_decision_inference(&self, _var: Variable, state:&Self::State) -> Option<Decision>{
        if let Some(model) = &self.ml_model {
            let output = self.perform_inference(&model,state);
            self.extract_decision_from_model_output(state,output)
        }
        else{
            None
        }
        
    }
}

/// In addition to a dynamic programming (DP) model of the problem you want to solve, 
/// the branch and bound with MDD algorithm (and thus ddo) requires that you provide
/// an additional relaxation allowing to control the maximum amount of space used by
/// the decision diagrams that are compiled. 
/// 
/// That relaxation requires two operations: one to merge several nodes into one 
/// merged node that acts as an over approximation of the other nodes. The second
/// operation is used to possibly offset some weight that would otherwise be lost 
/// to the arcs entering the newly created merged node.
/// 
/// The role of this very simple structure is simply to provide an implementation
/// of that relaxation.
/// 
/// # Note:
/// In addition to the aforementioned two operations, the KPRelax structure implements
/// an optional `fast_upper_bound` method. Which one provides a useful bound to 
/// prune some portions of the state-space as the decision diagrams are compiled.
/// (aka rough upper bound pruning).
pub struct KPRelax<'a>{pub pb: &'a Knapsack}
impl Relaxation for KPRelax<'_> {
    type State = KnapsackState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        states.max_by_key(|node| node.capacity).copied().unwrap()
    }

    fn relax(&self, _source: &Self::State, _dest: &Self::State, _merged: &Self::State, _decision: Decision, cost: isize) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let mut depth = state.depth;
        let mut max_profit = 0;
        let mut capacity = state.capacity;

        while capacity > 0 && depth < self.pb.profit.len() {
            let item = self.pb.order[depth];

            if capacity >= self.pb.weight[item] {
                max_profit += self.pb.profit[item];
                capacity -= self.pb.weight[item];
            } else {
                let item_ratio = capacity as f64 / self.pb.weight[item] as f64;
                let item_profit = item_ratio * self.pb.profit[item] as f64;
                max_profit += item_profit.floor() as isize;
                capacity = 0;
            }

            depth += 1;
        }

        max_profit
    }
}

/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct KPRanking;
impl StateRanking for KPRanking {
    type State = KnapsackState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.capacity.cmp(&b.capacity)
    }
}

/// Optionally, define dominance relations between states obtained throughout the search.
/// In this case, s1 dominates s2 if s1.capacity >= s2.capacity and s1 has a larger value than s2.
pub struct KPDominance;
impl Dominance for KPDominance {
    type State = KnapsackState;
    type Key = usize;

    fn get_key(&self, state: Arc<Self::State>) -> Option<Self::Key> {
        Some(state.depth)
    }

    fn nb_dimensions(&self, _state: &Self::State) -> usize {
        1
    }

    fn get_coordinate(&self, state: &Self::State, _: usize) -> isize {
        state.capacity as isize
    }

    fn use_value(&self) -> bool {
        true
    }
}

// #########################################################################################
// # THE INFORMATION BEYOND THIS LINE IS NOT DIRECTLY RELATED TO THE IMPLEMENTATION OF     #
// # A SOLVER BASED ON DDO. INSTEAD, THAT PORTION OF THE CODE CONTAINS GENERIC FUNCTION    #
// # THAT ARE USED TO READ AN INSTANCE FROM FILE, PROCESS COMMAND LINE ARGUMENTS, AND      #
// # THE MAIN FUNCTION. THESE ARE THUS NOT REQUIRED 'PER-SE', BUT I BELIEVE IT IS USEFUL   #
// # TO SHOW HOW IT CAN BE DONE IN AN EXAMPLE.                                             #
// #########################################################################################

/// This structure uses `clap-derive` annotations and define the arguments that can
/// be passed on to the executable solver.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the instance file
    fname: String,
    /// The number of concurrent threads
    #[clap(short, long, default_value = "8")]
    threads: usize,
    /// The maximum amount of time you would like this solver to run
    #[clap(short, long, default_value = "3000")]
    duration: u64,
    /// The maximum width of a layer when solving an instance. By default, it will allow
    /// as many nodes in a layer as there are unassigned variables in the global problem.
    #[clap(short, long)]
    width: Option<usize>,
    /// /// Whether or not to use clustering to split nodes. True if -c supplied. Uses ckmeans clustering.
    #[clap(short, long, action)]
    cluster: bool,
    /// Option to use ML model for restriction builidng
    /// Path to pb file for model
    #[clap(short, long, default_value = "")]
    model: String,
    /// Whether or not to write output to json file
    #[clap(short, long, action)]
    json_output: bool,
    /// Path to write output file to
    #[clap(short='x', long, default_value = "")]
    outfolder: String,
    /// Solver to use
    #[clap(short='s', long, default_value = "IR")]
    solver: String,
}

/// This enumeration simply groups the kind of errors that might occur when parsing a
/// knapsack instance from file. There can be io errors (file unavailable ?), format error
/// (e.g. the file is not a knapsack instance but contains the text of your next paper), 
/// or parse int errors (which are actually a variant of the format error since it tells 
/// you that the parser expected an integer number but got ... something else).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// There was an io related error
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    /// The parser expected to read something that was an integer but got some garbage
    #[error("parse int {0}")]
    ParseInt(#[from] ParseIntError),
    /// The file was not properly formatted.
    #[error("ill formed instance")]
    Format,
}

/// This function is used to read a knapsack instance from file. It returns either a
/// knapsack instance if everything went on well or an error describing the problem.
fn read_instance(args:&Args) -> Result<Knapsack, Error> {
    let f = File::open(&args.fname)?;
    let f = BufReader::new(f);
    let clustering = args.cluster;
    
    let mut is_first = true;
    let mut n = 0;
    let mut count = 0;
    let mut capa = 0;
    let mut profit = vec![];
    let mut weight = vec![];

    for line in f.lines() {
        let line = line?;
        if line.starts_with('c') {
            continue;
        }
        if is_first {
            is_first = false;
            let mut ab = line.split(' ');
            n = ab.next().ok_or(Error::Format)?.parse()?;
            capa = ab.next().ok_or(Error::Format)?.parse()?;
        } else {
            if count >= n {
                break;
            }
            let mut ab = line.split(' ');
            profit.push(ab.next().ok_or(Error::Format)?.parse()?);
            weight.push(ab.next().ok_or(Error::Format)?.parse()?);
            count += 1;
        }
    }
    let model: Option<TfModel> = if !args.model.is_empty(){Some(read_model(&args.model,"test_input".to_string(),"test_output".to_string()).unwrap())} 
                                else {None};
    Ok(Knapsack::new(capa, profit, weight, clustering, model))
}

pub fn read_model<P: AsRef<Path>>(model_path: P,input: String, output:String) -> Result<TfModel, Error>{
    Ok(TfModel::new(model_path,input,output))
}
/// An utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptive policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<T>(nb_vars: usize, w: Option<usize>) -> Box<dyn WidthHeuristic<T> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
    } else {
        Box::new(NbUnassignedWidth(nb_vars))
    }
}

/// This is your executable's entry point. It is the place where all the pieces are put together
/// to create a fast an effective solver for the knapsack problem.
fn main() {
    let args = Args::parse();
    let problem = read_instance(&args).unwrap();
    let relaxation= KPRelax{pb: &problem};
    let heuristic= KPRanking;
    let width = max_width(problem.nb_variables(), args.width);
    let dominance = SimpleDominanceChecker::new(KPDominance, problem.nb_variables());
    let cutoff = TimeBudget::new(Duration::from_secs(args.duration));//NoCutoff;
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));

    fn run_solve<T:Solver>(args:&Args, mut solver:T,) -> serde_json::Value{
        let start = Instant::now();
        let Completion {
            is_exact,
            best_value,
        } = solver.maximize();

        let duration = start.elapsed();
        let upper_bound = solver.best_upper_bound();
        let lower_bound = solver.best_lower_bound();
        let gap = solver.gap();
        let best_solution = solver.best_solution().map(|mut decisions| {
            decisions.sort_unstable_by_key(|d| d.variable.id());
            decisions.iter().map(|d| d.value).collect::<Vec<_>>()
        });

        let result = json!({
            "Duration": format!("{:.3}", duration.as_secs_f32()),
            "Objective":  format!("{}", best_value.unwrap_or(-1)),
            "Upper Bnd":  format!("{}", upper_bound),
            "Lower Bnd":  format!("{}", lower_bound),
            "Gap":        format!("{:.3}", gap),
            "Aborted":    format!("{}", !is_exact),
            "Cluster":    format!("{}", args.cluster),
            "Solver":    format!("{}", args.solver),
            "Width":    format!("{}", args.width.unwrap_or(0)),
            "Solution":   format!("{:?}", best_solution.unwrap_or_default())
        });

        result
    }

    // let mut solver = SeqIncrementalSolver::new(
    //     &problem,
    //     &relaxation,
    //     &heuristic,
    //     width.as_ref(),
    //     &dominance,
    //     &cutoff,
    //     &mut fringe,
    // );

    // let start = Instant::now();
    // let Completion{ is_exact, best_value } = solver.maximize();
    
    // let duration = start.elapsed();
    // let upper_bound = solver.best_upper_bound();
    // let lower_bound = solver.best_lower_bound();
    // let gap = solver.gap();
    // let best_solution  = solver.best_solution().map(|mut decisions|{
    //     decisions.sort_unstable_by_key(|d| d.variable.id());
    //     decisions.iter().map(|d| d.value).collect::<Vec<_>>()
    // });

    // println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    // println!("Objective:  {}",            best_value.unwrap_or(-1));
    // println!("Upper Bnd:  {}",            upper_bound);
    // println!("Lower Bnd:  {}",            lower_bound);
    // println!("Gap:        {:.3}",         gap);
    // println!("Aborted:    {}",            !is_exact);
    // println!("Solution:   {:?}",          best_solution.unwrap_or_default());

    let result = match args.solver.as_str() {
        "TD" => {
            let solver = TDCompile::new(
                &problem,
                &relaxation,
                &heuristic,
                width.as_ref(),
                &dominance,
                &cutoff,
                &mut fringe,
            );
            run_solve(&args,solver)
        },
        "IR" => {
            let solver = SeqIncrementalSolver::new(
                &problem,
                &relaxation,
                &heuristic,
                width.as_ref(),
                &dominance,
                &cutoff,
                &mut fringe,
            );
            run_solve(&args,solver)
        },
        "BB" => {
            let solver = SeqCachingSolverLel::new(
                &problem,
                &relaxation,
                &heuristic,
                width.as_ref(),
                &dominance,
                &cutoff,
                &mut fringe,
            );
            run_solve(&args,solver)
        },
        _ => panic!("suplied unknown solver")};
    
    
    println!("{}", result.to_string());
    if args.json_output{
        let mut outfile = args.outfolder.to_owned();
        let instance_name = if let Some(x) = &args.fname.split("/").collect::<Vec<_>>().last() {x} else {"_"};
        outfile.push_str(&instance_name);
        outfile.push_str(".json");
        fs::write(outfile,result.to_string()).expect("unable to write json");
    }
}
