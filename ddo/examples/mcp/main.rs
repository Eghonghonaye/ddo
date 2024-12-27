use std::{fs::{self, File}, time::{Duration, Instant}};

use clap::Parser;
use ddo::*;
use serde_json::json;

use crate::{graph::Graph, model::{Mcp, McpRanking}, relax::McpRelax};

mod graph;
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

fn main() {
    // let Params{file, width, duration} = Params::parse();
    let args = Params::parse();
    let graph = Graph::from(File::open(&args.file).expect("could not open file"));
    let problem = Mcp::from(graph);
    let relax = McpRelax::new(&problem);
    let rank = McpRanking;
    let width = max_width(&problem, args.width);
    let dominance = EmptyDominanceChecker::default();
    let cutoff = cutoff(args.duration);
    let mut fringe = NoDupFringe::new(MaxUB::new(&rank));

    // let mut solver = DefaultSolver::new(
    //     &problem, 
    //     &relax, 
    //     &rank, 
    //     width.as_ref(), 
    //     &dominance,
    //     cutoff.as_ref(), 
    //     &mut fringe,
    // );

    fn run_solve<T: Solver>(args: &Params, mut solver: T) -> serde_json::Value {
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
                &relax,
                &rank,
                width.as_ref(),
                &dominance,
                cutoff.as_ref(),
                &mut fringe,
                args.cluster_compile,
            );
            run_solve(&args, solver)
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
            run_solve(&args, solver)
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
            run_solve(&args, solver)
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