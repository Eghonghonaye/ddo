use crate::abstraction::constraints::{Satisfaction,Setup};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;

impl Setup{
    fn feasible(&self, 
        _instance: &Instance, 
        state: &State, 
        op: &OpId) -> bool{
            if self.op_b == *op{
                if state.est[self.op_b.as_usize()] > state.lst[self.op_a.as_usize()] + self.value{
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
            println!("before setup filter {:?}", options);
            for index in 0..instance.nops{
                if !self.feasible(instance,state,&OpId::new(index)) {
                        options.remove(index);
                        println!("after setup filter {:?}", options);
                    } 
            } 
        // options.retain(|op| self.feasible(instance,state,&op) == true);
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}




