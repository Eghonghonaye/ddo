use crate::abstraction::constraints::{Constraint,Satisfaction};
// use crate::constraints::{Constraint,Satisfaction,Deadline,Release,Processing,Setup,Precedence};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;


impl Satisfaction for Constraint{
    fn filter_set(&self,
        instance:&Instance, 
        state:&State,
        op: &OpId,
        options:& mut BitVector) {
        match self{
            Constraint::DeadlineCons(cons) => cons.filter_set(instance,state,op,options),
            Constraint::SetupCons(cons) => cons.filter_set(instance,state,op,options),
            Constraint::ReleaseCons(cons) => cons.filter_set(instance,state,op,options),
            Constraint::PrecedenceCons(cons) => cons.filter_set(instance,state,op,options),
            Constraint::ProcessingCons(cons) => cons.filter_set(instance,state,op,options),
            Constraint::AssignCons(cons) => cons.filter_set(instance,state,op,options)
        }
    }
    fn filter_edge(&self) -> bool{
        match self{
            Constraint::DeadlineCons(cons) => cons.filter_edge(),
            Constraint::SetupCons(cons) => cons.filter_edge(),
            Constraint::ReleaseCons(cons) => cons.filter_edge(),
            Constraint::PrecedenceCons(cons) => cons.filter_edge(),
            Constraint::ProcessingCons(cons) => cons.filter_edge(),
            Constraint::AssignCons(cons) => cons.filter_edge()
        }

    }
}