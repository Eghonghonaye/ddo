// traits of all constraints

use crate::abstraction::instance::{Instance,Machine,OpId};
use crate::bitvector::BitVector;
use crate::model::State;
use std::rc::Rc;

#[derive(Clone, PartialEq, Eq)]
pub enum Constraint{
    DeadlineCons(Deadline),
    SetupCons(Setup),
    ReleaseCons(Release),
    PrecedenceCons(Precedence),
    ProcessingCons(Processing),
    AssignCons(Assign),
    NoRepeatCons(NoRepeat)
}

// impl Constraint {
//     // tell us if the constrant is satisfied for a given partial solution
//     fn feasible()-> bool{
//         true
//     }
// }

pub trait Satisfaction {
    fn filter_set(&self,
        instance:&Instance,
        state: &State,
        op: &OpId,
        options:& mut BitVector);
    fn filter_edge(&self,) -> bool;
}

#[derive(Clone, PartialEq, Eq)]
pub struct Deadline {
    pub op_a: OpId,
    pub value: usize
}

#[derive(Clone, PartialEq, Eq)]
pub enum SetupType{
    PureSequence,
    MachineSequence,
    IndependentOps,
    IndependentMachines
}

#[derive(Clone, PartialEq, Eq)]
pub struct Setup {
    pub op_a: OpId,
    pub op_b: OpId,
    pub value: usize,
    pub setup_type: SetupType
}

#[derive(Clone, PartialEq, Eq)]
pub struct Assign {
    pub op_a: OpId,
    pub mach: Rc<Machine>
}

#[derive(Clone, PartialEq, Eq)]
pub struct Release {
    pub op_a: OpId,
    pub value: usize
}

#[derive(Clone, PartialEq, Eq)]
pub struct Precedence {
    pub op_a: OpId,
    pub op_b: OpId
}

#[derive(Clone, PartialEq, Eq)]
pub struct Processing {
    pub op_a: OpId,
    pub value: usize
}

#[derive(Clone, PartialEq, Eq)]
pub struct NoRepeat{
    pub op_a: OpId
}