use crate::abstraction::constraints::{Satisfaction,NoRepeat};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;


impl Satisfaction for NoRepeat{
    fn filter_set(&self, 
        instance: &Instance, 
        state: &State, 
        _op: &OpId,
        options: &mut BitVector) {
            println!("before no repeat filter {:?}", options);
            for index in 0..instance.nops{
                if state.def_scheduled.contains(index){
                        options.remove(index);
                        println!("after no repeat filter {:?}", options);
                    } 
            } 
            // options.retain(|op| self.feasible(instance,state,&op) == true);
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}