use crate::abstraction::constraints::{Satisfaction,Deadline};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;

impl Satisfaction for Deadline{
    fn filter_set(&self,
        instance:&Instance,
        state:&State,
        _op: &OpId,
        options:& mut BitVector){
            println!("after deadline filter {:?}", options);
            for index in 0..instance.nops{
                if state.est[index] > state.lst[index] || state.est[index] > self.value  {
                        options.remove(index);
                        println!("after deadline filter {:?}", options);
                    } 
            } 

        // // TODO: can we do this with iterator instead -- currently broken
        // let mut feasible_options = BitVector::new(instance.nops);
        // let x = feasible_options.iter().enumerate().map(|(index,_x)| 
        //                                 if state.est[index] <= state.lst[index] || state.est[index] <= self.value || state.est[index] <= self.value  
        //                                         {1usize;
        //                                             //feasible_options.insert(index)
        //                                         } 
        //                                 else {0usize;
        //                                     //feasible_options.remove(index)
        //                                 }).collect::<Vec<_>>();
        // feasible_options = BitVector::from_iter::<Vec<_>>(x);
        // options.intersection_inplace(&feasible_options);
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}

