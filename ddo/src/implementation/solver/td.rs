use std::clone::Clone;
use std::{hash::Hash, sync::Arc};

use crate::{
    Cache, CompilationInput, CompilationType, Completion, Cutoff, Decision, DecisionDiagram,
    DefaultMDDLEL, DominanceChecker, EmptyCache, Fringe, Problem, Reason, Relaxation, Solution,
    Solver, StateRanking, SubProblem, WidthHeuristic
};

pub struct TDSolver<
    'a,
    State,
    D = DefaultMDDLEL<State>,
    C = EmptyCache<State>,
> where
    D: DecisionDiagram<State = State> + Default,
    C: Cache<State = State> + Default,
{
    /// A reference to the problem being solved with branch-and-bound MDD
    problem: &'a (dyn Problem<State = State>),
    /// The relaxation used when a DD layer grows too large
    relaxation: &'a (dyn Relaxation<State = State>),
    /// The ranking heuristic used to discriminate the most promising from
    /// the least promising states
    value_ranking: &'a (dyn StateRanking<State = State>),
    /// The ranking heuristic used to discriminate the nodes to split first
    /// We aim to split nodes that will strengthen the relaxation first, can supply same
    /// ranking as value_ranking if you want but not necessary for this ranking to be based on
    /// contribution to objective
    split_ranking: &'a (dyn StateRanking<State = State>),
    /// The maximum width heuristic used to enforce a given maximum memory
    /// usage when compiling mdds
    width_heu: &'a (dyn WidthHeuristic<State>),
    /// A cutoff heuristic meant to decide when to stop the resolution of
    /// a given problem.
    cutoff: &'a (dyn Cutoff),

    /// This is the fringe: the set of nodes that must still be explored before
    /// the problem can be considered 'solved'.
    ///
    /// # Note:
    /// This fringe orders the nodes by upper bound (so the highest ub is going
    /// to pop first). So, it is guaranteed that the upper bound of the first
    /// node being popped is an upper bound on the value reachable by exploring
    /// any of the nodes remaining on the fringe. As a consequence, the
    /// exploration can be stopped as soon as a node with an ub <= current best
    /// lower bound is popped.
    fringe: &'a mut (dyn Fringe<State = State>),
    /// This is a counter that tracks the number of nodes that have effectively
    /// been explored. That is, the number of nodes that have been popped from
    /// the fringe, and for which a restricted and relaxed mdd have been developed.
    explored: usize,
    /// This is a counter of the number of nodes in the fringe, for each level of the model
    open_by_layer: Vec<usize>,
    /// This is the index of the first level above which there are no nodes in the fringe
    first_active_layer: usize,
    /// This is the value of the best known lower bound.
    best_lb: isize,
    /// This is the value of the best known upper bound.
    best_ub: isize,
    /// If set, this keeps the info about the best solution so far.
    best_sol: Option<Vec<Decision>>,
    /// If we decide not to go through a complete proof of optimality, this is
    /// the reason why we took that decision.
    abort_proof: Option<Reason>,

    /// This is just a marker that allows us to remember the exact type of the
    /// mdds to be instantiated.
    mdd: D,

    /// Data structure containing info about past compilations used to prune the search
    cache: C,
    dominance: &'a (dyn DominanceChecker<State = State>),
}

impl<'a, State, D, C> TDSolver<'a, State, D, C>
where
    State: Eq + Hash + Clone,
    D: DecisionDiagram<State = State> + Default,
    C: Cache<State = State> + Default,
{
    pub fn new(
        problem: &'a (dyn Problem<State = State>),
        relaxation: &'a (dyn Relaxation<State = State>),
        ranking: &'a (dyn StateRanking<State = State>),
        width: &'a (dyn WidthHeuristic<State>),
        dominance: &'a (dyn DominanceChecker<State = State>),
        cutoff: &'a (dyn Cutoff),
        fringe: &'a mut (dyn Fringe<State = State>),
    ) -> Self {
        Self::custom(
            problem, relaxation, ranking, width, dominance, cutoff, fringe,
        )
    }

    pub fn custom(
        problem: &'a (dyn Problem<State = State>),
        relaxation: &'a (dyn Relaxation<State = State>),
        ranking: &'a (dyn StateRanking<State = State>),
        width_heu: &'a (dyn WidthHeuristic<State>),
        dominance: &'a (dyn DominanceChecker<State = State>),
        cutoff: &'a (dyn Cutoff),
        fringe: &'a mut (dyn Fringe<State = State>),
    ) -> Self {
        TDSolver {
            problem,
            relaxation,
            value_ranking: ranking,
            split_ranking: ranking,
            width_heu,
            cutoff,
            //
            best_sol: None,
            best_lb: isize::MIN,
            best_ub: isize::MAX,
            fringe,
            explored: 0,
            open_by_layer: vec![0; problem.nb_variables() + 1],
            first_active_layer: 0,
            abort_proof: None,
            mdd: D::default(),
            cache: C::default(),
            dominance,
        }
    }

    fn initialize(&mut self) -> Result<Completion, Reason> {
        // create root
        let root = self.root_node();
        let best_lb = self.best_lb;

        let width = self.width_heu.max_width(&root);
        let compilation = CompilationInput {
            comp_type: CompilationType::Relaxed,
            comp_strategy: crate::CompilationStrategy::TopDown,
            max_width: width,
            problem: self.problem,
            relaxation: self.relaxation,
            ranking: self.value_ranking,
            cutoff: self.cutoff,
            cache: &self.cache,
            dominance: self.dominance,
            residual: &root,
            //
            best_lb,
            binary_split: false,
        };

        // compile initial narrow width diagram
        let completion = self.mdd.compile(&compilation)?;

        //FIXME should return soemthing that tells me whether the initial diagram was successfully compiled or not
        Ok(completion)
    }

    fn root_node(&self) -> SubProblem<State> {
        SubProblem {
            state: Arc::new(self.problem.initial_state()),
            value: self.problem.initial_value(),
            path: vec![],
            ub: isize::MAX,
            depth: 0,
        }
    }

    fn abort_search(&mut self, reason: Reason) {
        self.abort_proof = Some(reason);
        self.fringe.clear();
        self.cache.clear();
    }

    /// This private method updates the shared best known node and lower bound in
    /// case the best value of the current `mdd` expansion improves the current
    /// bounds.
    fn maybe_update_best(&mut self) {
        let dd_best_value = self.mdd.best_exact_value().unwrap_or(isize::MIN);
        if dd_best_value > self.best_lb {
            self.best_lb = dd_best_value;
            self.best_sol = self.mdd.best_exact_solution();
        }
        let dd_upper_bound = self.mdd.best_value().unwrap_or(isize::MIN);
        if dd_upper_bound < self.best_ub {
            self.best_ub = dd_upper_bound;
        }
    }
}

impl<'a, State, D, C> Solver
    for TDSolver<'a, State, D, C>
where
    State: Eq + PartialEq + Hash + Clone,
    D: DecisionDiagram<State = State> + Default,
    C: Cache<State = State> + Default,
{
    /// This method orders the solver to search for the optimal solution among
    /// all possibilities. It returns a structure standing for the outcome of
    /// the attempted maximization. Such a `Completion` may either be marked
    /// **exact** if the maximization has been carried out until optimality was
    /// proved. Or it can be inexact, in which case it means that the
    /// maximization process was stopped because of the satisfaction of some
    /// cutoff criterion.
    ///
    /// Along with the `is_exact` exact flag, the completion provides an
    /// optional `best_value` of the maximization problem. Four cases are thus
    /// to be distinguished:
    ///
    /// * When the `is_exact` flag is true, and a `best_value` is present: the
    ///   `best_value` is the maximum value of the objective function.
    /// * When the `is_exact` flag is false and a `best_value` is present, it
    ///   is the best value of the objective function that was known at the time
    ///   of cutoff.
    /// * When the `is_exact` flag is true, and no `best_value` is present: it
    ///   means that the problem admits no feasible solution (UNSAT).
    /// * When the `is_exact` flag is false and no `best_value` is present: it
    ///   simply means that no feasible solution has been found before the
    ///   cutoff occurred.
    ///
    ///
    ///
    fn maximize(&mut self) -> Completion {
        let outcome = self.initialize();
        if let Err(reason) = outcome {
            self.abort_search(reason);

            if let Some(sol) = self.best_sol.as_mut() {
                sol.sort_unstable_by_key(|d| d.variable.0)
            }
            return Completion {
                is_exact: self.abort_proof.is_none(),
                best_value: self.best_sol.as_ref().map(|_| self.best_lb),
            };
        }
        self.maybe_update_best();

        if let Some(sol) = self.best_sol.as_mut() {
            sol.sort_unstable_by_key(|d| d.variable.0)
        }
        Completion {
            is_exact: self.abort_proof.is_none(),
            best_value: self.best_sol.as_ref().map(|_| self.best_lb),
        }
    }

    /// Returns the best solution that has been identified for this problem.
    /// /// This method returns the best solution to the optimization problem.
    /// That is, it returns the vector of decision which maximizes the value
    /// of the objective function (sum of transition costs + initial value).
    /// It returns `None` when the problem admits no feasible solution.
    fn best_solution(&self) -> Option<Vec<Decision>> {
        self.best_sol.clone()
    }

    /// Returns the value of the best solution that has been identified for
    /// this problem.
    ///  /// This method returns the value of the objective function for the best
    /// solution that has been found. It returns `None` when no solution exists
    /// to the problem.
    fn best_value(&self) -> Option<isize> {
        self.best_sol.as_ref().map(|_sol| self.best_lb)
    }

    /// Returns the value of the best lower bound that has been identified for
    /// this problem.
    /// /// Returns the best lower bound that has been identified so far.
    /// In case where no solution has been found, it should return the minimum
    /// value that fits within an isize (-inf).
    fn best_lower_bound(&self) -> isize {
        self.best_lb
    }

    /// Returns the value of the best upper bound that has been identified for
    /// this problem.
    /// /// Returns the tightest upper bound that can be guaranteed so far.
    /// In case where no upper bound has been computed, it should return the
    /// maximum value that fits within an isize (+inf).
    fn best_upper_bound(&self) -> isize {
        self.best_ub
    }

    /// Sets a primal (best known value and solution) of the problem.
    fn set_primal(&mut self, value: isize, solution: Solution) {
        if value > self.best_lb {
            self.best_sol = Some(solution);
            self.best_lb = value;
        }
    }
}