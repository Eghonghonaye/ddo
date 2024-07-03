use crate::abstraction::constraints::{Satisfaction,Precedence};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;


impl Precedence{
    fn feasible(&self, 
        _instance: &Instance, 
        state: &State, 
        op: &OpId) -> bool{
            if self.op_b.id == *op{
                if !state.maybe_scheduled.contains(self.op_a.id.as_usize()){
                    return false;
                }
            }  
            return true;
    }
}


impl Satisfaction for Precedence{
    fn filter_set(&self, 
        instance: &Instance, 
        state: &State, 
        _op: &OpId,
        options: &mut BitVector) {
            for index in 0..options.capacity(){
                if !self.feasible(instance,state,&OpId::new(index)){
                        options.remove(index);
                    } 
            } 
            // options.retain(|op| self.feasible(instance,state,&op) == true);
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}