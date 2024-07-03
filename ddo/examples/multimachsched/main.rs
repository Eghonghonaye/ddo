use std::time::{Duration, Instant};

use clap::Parser;
use ddo::*;

mod model;

// use crate::instance::Instance;
// mod instance;
// mod constraints;

mod abstraction;
mod implementation;
mod utils;

// pub use abstraction::*;
// pub use implementation::*;
pub use utils::*;
use model::{Mms, MmsRelax};
use model::MmsRanking;

use crate::abstraction::instance::Instance;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Params {
    /// the instance file
    #[arg(short, long)]
    file: String,
    /// maximum width in a layer
    #[arg(short, long)]
    width: Option<usize>,
    /// max time to find the solution
    #[arg(short, long)]
    timeout: Option<u64>,
}


/// An utility function to return an max width heuristic that can either be a fixed width
/// policy (if w is fixed) or an adaptive policy returning the number of unassigned variables
/// in the overall problem.
fn max_width<T,X>(nb_vars: usize, w: Option<usize>) -> Box<dyn WidthHeuristic<T,X> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
    } else {
        Box::new(NbUnassignedWidth(nb_vars))
    }
}


fn main(){
    let Params{file, width, timeout} = Params::parse();

    let instance: Instance = Instance::from_file(file);
    let problem: Mms = Mms::initialise(instance);

    let relaxation= MmsRelax{problem: &problem};
    let heuristic= MmsRanking;
    let width = max_width(problem.nb_variables(), width);

    // let dominance = SimpleDominanceChecker::new(KPDominance, problem.nb_variables());
    let dominance = EmptyDominanceChecker::default(); // dummy dominance checker that does nothing
    let cutoff = TimeBudget::new(Duration::from_secs(15));//NoCutoff;
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));

    let mut solver = SeqNoCachingSolverLel::new(
        &problem, 
        &relaxation, 
        &heuristic, 
        width.as_ref(), 
        &dominance,
        &cutoff, 
        &mut fringe
    );

    let start = Instant::now();
    let Completion{ is_exact, best_value } = solver.maximize();
    
    let duration = start.elapsed();
    let upper_bound = solver.best_upper_bound();
    let lower_bound = solver.best_lower_bound();
    let gap = solver.gap();
    let best_solution  = solver.best_solution().map(|mut decisions|{
        decisions.sort_unstable_by_key(|d| d.variable.id());
        decisions.iter().map(|d| d.value).collect::<Vec<_>>()
    });

    println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    println!("Objective:  {}",            best_value.unwrap_or(-1));
    println!("Upper Bnd:  {}",            upper_bound);
    println!("Lower Bnd:  {}",            lower_bound);
    println!("Gap:        {:.3}",         gap);
    println!("Aborted:    {}",            !is_exact);
    println!("Solution:   {:?}",          best_solution.unwrap_or_default());
}