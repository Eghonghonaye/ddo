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

//! This module defines the types used to encode the state of a node in the 
//! SRFLP problem.

use std::hash::Hash;

use smallbitset::Set64;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SrflpState {
    /// These are the departments that need to be placed
    pub must_place : Set64,
    /// These are the departments that maybe need to be placed
    pub maybe_place: Option<Set64>,
    /// Total flow from fixed departments to each free department
    pub cut: Vec<isize>,
    /// This is the 'depth' in the arrangement, the number of departments that have already been placed
    pub depth: usize
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SrflpDecisionState;
