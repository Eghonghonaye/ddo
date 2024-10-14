use crate::abstraction::constraints::{Satisfaction,Assign};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;

impl Satisfaction for Assign{
    fn filter_set(&self,
        _instance:&Instance,
        _state:&State,
         _op: &OpId,
        _options:& mut BitVector){
        // currently checks nothing as it is assumed operation values are calculated on asisgned machine
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}