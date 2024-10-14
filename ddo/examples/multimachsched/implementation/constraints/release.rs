use crate::abstraction::constraints::{Satisfaction,Release};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;

impl Satisfaction for Release{
    fn filter_set(&self,
        instance:&Instance,
        state:&State,
        _op: &OpId,
        options:& mut BitVector){
            println!("before release filter {:?}", options);
            for index in 0..instance.nops{
                if state.est[index] < self.value {
                        options.remove(index);
                        println!("after release filter {:?}", options);
                    } 
            } 
        // options.retain(|operation| state.est[operation.as_usize()] >= self.value);
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}


