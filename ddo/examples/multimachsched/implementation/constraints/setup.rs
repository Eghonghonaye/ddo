use crate::abstraction::constraints::{Satisfaction,Setup};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;

impl Setup{
    fn feasible(&self, 
        _instance: &Instance, 
        state: &State, 
        op: &OpId) -> bool{
            if self.op_b.id == *op{
                if state.est[self.op_b.id.as_usize()] >= state.lst[self.op_a.id.as_usize()] + self.value{
                    return false;
                }
            }  
            return true;
    }
}

impl Satisfaction for Setup{
    fn filter_set(&self,
        instance:&Instance,
        state:&State,
        _op: &OpId,
        options:& mut BitVector){
            for index in 0..options.capacity(){
                if !self.feasible(instance,state,&OpId::new(index)) {
                        options.remove(index);
                    } 
            } 
        // options.retain(|op| self.feasible(instance,state,&op) == true);
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}




