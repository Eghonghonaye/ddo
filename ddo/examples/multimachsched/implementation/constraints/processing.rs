use crate::abstraction::constraints::{Satisfaction,Processing};
use crate::abstraction::instance::{Instance,OpId};
use crate::bitvector::BitVector;
use crate::model::State;

impl Satisfaction for Processing{
    fn filter_set(&self,
        _instance:&Instance,
        _state:&State,
        _op: &OpId,
        _options:& mut BitVector){
        // currently checks nothing as it is assumed est is calculated with this in mind
        // check should check gap between consecutive ops on same m/c
    }

    fn filter_edge(&self,) -> bool{
        true
    }
}