use std::{fs, time::{Duration, Instant}};

use clap::Parser;
use ddo::*;
use model::{f, t};
use serde_json::json;

use crate::{heuristics::Max2SatRanking, model::{Max2Sat, v}, relax::Max2SatRelax, data::read_instance};

mod errors;
mod heuristics;
mod data;
mod model;
mod relax;
#[cfg(test)]
mod tests;

/// Solve max2sat instance
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Params {
    /// the instance file
    // #[arg(short, long)]
    file: String,
    /// maximum width in a layer
    #[arg(short, long)]
    width: Option<usize>,
    /// max time to find the solution
    #[arg(short, long)]
    duration: Option<u64>,
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
    /// Compile top down by clustering for mwege
    #[clap(short = 'k', long, action)]
    cluster_compile: bool,
}

fn main() {
    // let Params{file, width, duration} = Params::parse();
    let args = Params::parse();
    let problem = Max2Sat::new(read_instance(&args.file).unwrap());
    let relax = Max2SatRelax(&problem);
    let rank = Max2SatRanking;
    let width = max_width(&problem, args.width);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = cutoff(args.duration);
    let mut fringe = NoDupFringe::new(MaxUB::new(&Max2SatRanking));

    // let mut solver = DefaultSolver::new(
    //     &problem, 
    //     &relax, 
    //     &rank, 
    //     width.as_ref(), 
    //     &dominance,
    //     cutoff.as_ref(), 
    //     &mut fringe,
    // );

    // println!("Duration:   {:.3} seconds", duration.as_secs_f32());
    // println!("Objective:  {}",            best_value.unwrap_or(-1));
    // println!("Upper Bnd:  {}",            upper_bound);
    // println!("Lower Bnd:  {}",            lower_bound);
    // println!("Gap:        {:.3}",         gap);
    // println!("Aborted:    {}",            !is_exact);
    // println!("Cost:       {:?}",          solution_cost(&problem, &solver.best_solution()));
    // println!("Solution:   {:?}",          best_solution.unwrap_or_default());

    fn run_solve<T:Solver>(args:&Params, problem:&Max2Sat, mut solver:T,) -> serde_json::Value{
        let start = Instant::now();
        let Completion{ is_exact, best_value } = solver.maximize();
        
        let duration = start.elapsed();
        let upper_bound = solver.best_upper_bound();
        let lower_bound = solver.best_lower_bound();
        let gap = solver.gap();
        let best_solution  = solver.best_solution().map(|mut decisions|{
            decisions.sort_unstable_by_key(|d| d.variable.id());
            decisions.iter().map(|d| v(d.variable) * d.value).collect::<Vec<_>>()
        });

        let result = json!({
            "Duration": format!("{:.3}", duration.as_secs_f32()),
            "Objective":  format!("{}", solution_cost(&problem, &solver.best_solution())),
            "Upper Bnd":  format!("{}", upper_bound),
            "Lower Bnd":  format!("{}", lower_bound),
            "Gap":        format!("{:.3}", gap),
            "Aborted":    format!("{}", !is_exact),
            "Refine Cluster":    format!("{}", args.cluster),
            "Compile Cluster":    format!("{}", args.cluster_compile),
            "Binary Split":    format!("{}", args.binary_split),
            "Solver":    format!("{}", args.solver),
            "Width":    format!("{}", args.width.unwrap_or(0)),
            "Solution":   format!("{:?}", best_solution.unwrap_or_default())
        });

        result
    }

    let result = match args.solver.as_str() {
        "TD" => {
            let solver = TDCompile::new(
                &problem,
                &relax,
                &rank,
                width.as_ref(),
                &dominance,
                cutoff.as_ref(),
                &mut fringe,
                args.cluster_compile,
                    );
            run_solve(&args, &problem, solver)
        }
        "IR" => {
            let solver = SeqIncrementalSolver::new(
                &problem,
                &relax,
                &rank,
                width.as_ref(),
                &dominance,
                cutoff.as_ref(),
                &mut fringe,
                args.binary_split,
                args.cluster_compile,
            );
            run_solve(&args, &problem, solver)
        }
        "BB" => {
            let solver = SeqCachingSolverLel::new(
                &problem,
                &relax,
                &rank,
                width.as_ref(),
                &dominance,
                cutoff.as_ref(),
                &mut fringe,
            );
            run_solve(&args, &problem, solver)
        }
        _ => panic!("suplied unknown solver"),
    };

    println!("{}", result.to_string());
    if args.json_output {
        let mut outfile = args.outfolder.to_owned();
        let instance_name = if let Some(x) = &args.file.split("/").collect::<Vec<_>>().last() {
            x
        } else {
            "_"
        };
        outfile.push_str(&instance_name);
        outfile.push_str(".json");
        fs::write(outfile, result.to_string()).expect("unable to write json");
    }
}

fn cutoff(duration: Option<u64>) -> Box<dyn Cutoff + Send + Sync> {
    if let Some(t) = duration {
        Box::new(TimeBudget::new(Duration::from_secs(t)))
    } else {
        Box::new(NoCutoff)
    }
}
fn max_width<P: Problem>(p: &P, w: Option<usize>) -> Box<dyn WidthHeuristic<P::State> + Send + Sync> {
    if let Some(w) = w {
        Box::new(FixedWidth(w))
    } else {
        Box::new(NbUnassignedWidth(p.nb_variables()))
    }
}

fn solution_cost(pb: &Max2Sat, solution: &Option<Vec<Decision>>) -> isize {
    if let Some(sol) = solution {
        let n = pb.nb_vars;
        let mut model = vec![0; n];
        for d in sol.iter() {
            model[d.variable.id()] = d.value;
        }

        let mut cost = 0;
        for i in 0..n {
            for j in i..n {
                if model[i] == 1 && model[j] == 1 {
                    cost += pb.weight(f(Variable(i)), f(Variable(j)))
                }
                if model[i] ==-1 && model[j] ==-1 {
                    cost += pb.weight(t(Variable(i)), t(Variable(j)))
                }
                if model[i] == 1 && model[j] ==-1 {
                    cost += pb.weight(f(Variable(i)), t(Variable(j)))
                }
                if model[i] ==-1 && model[j] == 1 {
                    cost += pb.weight(t(Variable(i)), f(Variable(j)))
                }
            }
        }
        cost
    } else {
        0
    }
}