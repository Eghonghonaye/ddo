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

//! This example uses ddo to solve the Sequential Ordering Problem
//! Instances can be downloaded from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/sop/

use std::{fs, time::{Duration, Instant}};

use clap::Parser;
use ddo::*;
use heuristics::SopWidth;
use serde_json::json;
use smallbitset::Set256;

use crate::{io_utils::read_instance, relax::SopRelax, heuristics::SopRanking, model::Sop};

type BitSet = Set256;

mod state;
mod model;
mod relax;
mod heuristics;
mod io_utils;

#[cfg(test)]
mod tests;

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
    #[clap(short, long)]
    duration: Option<u64>,
    /// The maximum number of nodes per layer
    #[clap(short, long)]
    width: Option<usize>,
    /// /// Whether or not to use clustering to split nodes. True if -c supplied. Uses ckmeans clustering.
    #[clap(short, long, action)]
    cluster: bool,
    /// Whether or not to write output to json file
    #[clap(short, long, action)]
    json_output: bool,
    /// Path to write output file to
    #[clap(short='x', long, default_value = "")]
    outfolder: String,
    /// Solver to use
    #[clap(short='s', long, default_value = "IR")]
    solver: String,
    /// Have nodes split into two instead of a whole layer split
    #[clap(short = 'b', long, action)]
    binary_split: bool,
}

/// An utility function to return a cutoff heuristic that can either be a time budget policy
/// (if timeout is fixed) or no cutoff policy.
fn cutoff(timeout: Option<u64>) -> Box<dyn Cutoff + Send + Sync> {
    if let Some(t) = timeout {
        Box::new(TimeBudget::new(Duration::from_secs(t)))
    } else {
        Box::new(NoCutoff)
    }
}

/// This is your executable's entry point. It is the place where all the pieces are put together
/// to create a fast an effective solver for the knapsack problem.
// fn main() {
//     let args = Args::parse();
//     let fname = &args.fname;
//     let instance = read_instance(fname).unwrap();
//     let problem = Sop::new(instance,args.cluster);
//     let relaxation = SopRelax::new(&problem);
//     let ranking = SopRanking;

//     let width = SopWidth::new(problem.nb_variables(), args.width.unwrap_or(1));
//     let dominance = EmptyDominanceChecker::default();
//     let cutoff = cutoff(args.duration);
//     let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

//     let mut solver = DefaultCachingSolver::custom(
//         &problem, 
//         &relaxation, 
//         &ranking, 
//         &width, 
//         &dominance,
//         cutoff.as_ref(), 
//         &mut fringe,
//         args.threads,
//     );

//     let start = Instant::now();
//     let Completion{ is_exact, best_value } = solver.maximize();
    
//     let duration = start.elapsed();
//     let upper_bound = - solver.best_upper_bound();
//     let lower_bound = - solver.best_lower_bound();
//     let gap = solver.gap();
//     let best_solution = solver.best_solution().unwrap_or_default()
//         .iter().map(|d| d.value).collect::<Vec<isize>>();
    
//     println!("Duration:   {:.3} seconds", duration.as_secs_f32());
//     println!("Objective:  {}",            best_value.map(|x| -x).unwrap_or(-1));
//     println!("Upper Bnd:  {}",            upper_bound);
//     println!("Lower Bnd:  {}",            lower_bound);
//     println!("Gap:        {:.3}",         gap);
//     println!("Aborted:    {}",            !is_exact);
//     println!("Solution:   {:?}",          best_solution);
// }


fn main() {
    let args = Args::parse();
    let fname = &args.fname;
    let instance = read_instance(fname).unwrap();
    let problem = Sop::new(instance,args.cluster);
    let relaxation = SopRelax::new(&problem);
    let ranking = SopRanking;

    let width = SopWidth::new(problem.nb_variables(), args.width.unwrap_or(1));
    let dominance = EmptyDominanceChecker::default();
    let cutoff = cutoff(args.duration);
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

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
        let best_solution = solver.best_solution().unwrap_or_default()
        .iter().map(|d| d.value).collect::<Vec<isize>>();

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
            "Solution":   format!("{:?}", best_solution)
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
                &ranking,
                &width,
                &dominance,
                cutoff.as_ref(),
                &mut fringe,
            );
            run_solve(&args,solver)
        },
        "IR" => {
            let solver = SeqIncrementalSolver::new(
                &problem,
                &relaxation,
                &ranking,
                &width,
                &dominance,
                cutoff.as_ref(),
                &mut fringe,
                args.binary_split,
            );
            run_solve(&args,solver)
        },
        "BB" => {
            let solver = SeqCachingSolverLel::new(
                &problem,
                &relaxation,
                &ranking,
                &width,
                &dominance,
                cutoff.as_ref(),
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
