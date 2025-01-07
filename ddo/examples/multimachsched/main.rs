use std::fs;
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
use model::MmsRanking;
use model::{Mms, MmsRelax};
use serde_json::{json, to_string_pretty};
pub use utils::*;

use crate::abstraction::instance::Instance;
use crate::abstraction::instance::OpId;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Params {
    /// the instance file
    file: String,
    /// The maximum amount of time you would like this solver to run
    #[clap(short, long, default_value = "30")]
    duration: u64,
    /// The number of concurrent threads
    #[clap(short, long, default_value = "2")]
    threads: usize,
    /// maximum width in a layer
    #[arg(short, long)]
    width: Option<usize>,
    /// /// Whether or not to use clustering to split nodes. True if -c supplied. Uses ckmeans clustering.
    #[clap(short, long, action)]
    cluster: bool,
    /// /// Whether or not to use dominance.
    #[clap(long, action)]
    dominance: bool,
    /// /// Whether or not to use fast upper bound.
    #[clap(long, action)]
    rub: bool,
    /// /// Whether or not to use variable ordering.
    #[clap(long, action)]
    variable_order: bool,
    /// Option to use ML model for restriction builidng
    /// Path to pb file for model
    #[clap(short, long, default_value = "")]
    model: String,
    /// Whether or not to write output to json file
    #[clap(short, long, action)]
    json_output: bool,
    /// Path to write output file to
    #[clap(short = 'x', long, default_value = "")]
    outfolder: String,
    /// Solver to use
    #[clap(short = 's', long, default_value = "IR")]
    solver: String,
    /// Have nodes split into two instead of a whole layer split
    #[clap(short = 'b', long, action)]
    binary_split: bool,
    /// Compile top down by clustering for mwege
    #[clap(short = 'k', long, action)]
    cluster_compile: bool,
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

fn main() {
    // let Params {
    //     file,
    //     duration,
    //     threads,
    //     width,
    //     cluster,
    // } = Params::parse();
    let args = Params::parse();

    let instance: Instance = Instance::from_file(&args.file);
    let problem: Mms = Mms::initialise(instance, args.cluster);

    let relaxation = MmsRelax { problem: &problem };
    let heuristic = MmsRanking;
    let width = max_width(problem.nb_variables(), args.width);

    let dominance = EmptyDominanceChecker::default(); // dummy dominance checker that does nothing
    let no_dominance = EmptyDominanceChecker::default();
    let cutoff = TimeBudget::new(Duration::from_secs(args.duration)); //NoCutoff;
    let mut fringe = SimpleFringe::new(MaxUB::new(&heuristic));

    fn run_solve<T: Solver>(args: &Params, problem:&Mms, mut solver: T) -> serde_json::Value {   
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
            decisions.iter().map(|d| problem.instance.ops[&OpId::new(d.value as usize)].name.clone()).collect::<Vec<_>>()
        });
        let merge_quality = solver.merge_quality();


        let result = json!({
            "Duration": format!("{:.3}", duration.as_secs_f32()),
            "Objective":  format!("{}", best_value.unwrap_or(-1)),
            "Upper Bnd":  format!("{}", upper_bound),
            "Lower Bnd":  format!("{}", lower_bound),
            "Gap":        format!("{:.3}", gap),
            "Aborted":    format!("{}", !is_exact),
            "Refine Cluster":    format!("{}", args.cluster),
            "Compile Cluster":    format!("{}", args.cluster_compile),
            "Binary Split":    format!("{}", args.binary_split),
            "Dominance":    format!("{}", args.dominance),
            "MergeQuality":    format!("{:.3}", merge_quality),
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
                &relaxation,
                &heuristic,
                width.as_ref(),
                if args.dominance{&dominance} else{&no_dominance},
                &cutoff,
                &mut fringe,
                args.cluster_compile,
            );
            run_solve(&args, &problem, solver)
        }
        "IR" => {
            let solver = SeqIncrementalSolver::new(
                &problem,
                &relaxation,
                &heuristic,
                width.as_ref(),
                if args.dominance{&dominance} else{&no_dominance},
                &cutoff,
                &mut fringe,
                args.binary_split,
                args.cluster_compile,
            );
            run_solve(&args, &problem, solver)
        }
        "BB" => {
            let solver = SeqCachingSolverLel::new(
                &problem,
                &relaxation,
                &heuristic,
                width.as_ref(),
                if args.dominance{&dominance} else{&no_dominance},
                &cutoff,
                &mut fringe,
            );

            // let solver = DefaultCachingSolver::custom(
            //     &problem, 
            //     &relaxation, 
            //     &heuristic, 
            //     width.as_ref(), 
            //     if args.dominance{&dominance} else{&no_dominance},
            //     &cutoff, 
            //     &mut fringe,
            //     args.threads,
            // );
            run_solve(&args, &problem, solver)
        }
        _ => panic!("suplied unknown solver"),
    };

    println!("{}", to_string_pretty(&result).unwrap());
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
