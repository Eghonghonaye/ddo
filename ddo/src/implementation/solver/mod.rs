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

//! This module provide the solver implementation.
mod incremental;
mod td;
mod parallel;
mod sequential;
pub use incremental::*;
pub use parallel::*;
pub use sequential::*;
pub use td::*;

use crate::{DefaultMDDFC, DefaultMDDLEL, DefaultMappedLEL, EmptyCache, Pooled, SimpleCache};

/// A type alias to emphasize that this is the solver that should be used by default.
pub type DefaultSolver<'a, State, DecisionState> = ParNoCachingSolverLel<'a, State, DecisionState>;
pub type DefaultCachingSolver<'a, State, DecisionState> =
    ParCachingSolverFc<'a, State, DecisionState>;

pub type ParNoCachingSolverLel<'a, State, DecisionState> = ParallelSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDLEL<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;
pub type ParNoCachingSolverFc<'a, State, DecisionState> = ParallelSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDFC<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;
pub type ParNoCachingSolverPooled<'a, State, DecisionState> = ParallelSolver<
    'a,
    State,
    DecisionState,
    Pooled<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;

pub type ParCachingSolverLel<'a, State, DecisionState> = ParallelSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDLEL<State, DecisionState>,
    SimpleCache<State, DecisionState>,
>;
pub type ParCachingSolverFc<'a, State, DecisionState> = ParallelSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDFC<State, DecisionState>,
    SimpleCache<State, DecisionState>,
>;
pub type ParCachingSolverPooled<'a, State, DecisionState> = ParallelSolver<
    'a,
    State,
    DecisionState,
    Pooled<State, DecisionState>,
    SimpleCache<State, DecisionState>,
>;

pub type SeqNoCachingSolverLel<'a, State, DecisionState> = SequentialSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDLEL<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;
pub type SeqNoCachingSolverFc<'a, State, DecisionState> = SequentialSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDFC<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;
pub type SeqNoCachingSolverPooled<'a, State, DecisionState> = SequentialSolver<
    'a,
    State,
    DecisionState,
    Pooled<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;

pub type SeqCachingSolverLel<'a, State, DecisionState> = SequentialSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDLEL<State, DecisionState>,
    SimpleCache<State, DecisionState>,
>;
pub type SeqCachingSolverFc<'a, State, DecisionState> = SequentialSolver<
    'a,
    State,
    DecisionState,
    DefaultMDDFC<State, DecisionState>,
    SimpleCache<State, DecisionState>,
>;
pub type SeqCachingSolverPooled<'a, State, DecisionState> = SequentialSolver<
    'a,
    State,
    DecisionState,
    Pooled<State, DecisionState>,
    SimpleCache<State, DecisionState>,
>;

pub type SeqIncrementalSolver<'a, State, DecisionState> = IncrementalSolver<
    'a,
    State,
    DecisionState,
    DefaultMappedLEL<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;

pub type TDCompile<'a, State, DecisionState> = TDSolver<
    'a,
    State,
    DecisionState,
    DefaultMappedLEL<State, DecisionState>,
    EmptyCache<State, DecisionState>,
>;
