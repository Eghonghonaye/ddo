// problem instance
use crate::abstraction::constraints::Constraint;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct OpId(usize);
impl OpId{
    pub fn new(id:usize) -> Self {
        OpId(id)
    }
    pub fn as_isize(&self) -> isize{
        self.0 as isize
    }
    pub fn as_usize(&self) -> usize{
        self.0
    }
    pub fn as_string(&self) -> String{
        format!("{}",self.0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MId(usize);
impl MId{
    pub fn new(id:usize) -> Self {
        MId(id)
    }
    pub fn max() -> Self{
        MId(usize::MAX)
    }
    pub fn as_usize(&self) -> usize{
        self.0
    }
    pub fn as_string(&self) -> String{
        format!("{}",self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Operation{
    pub name: String,
    pub id : OpId,
    pub release: usize,
    pub processing: usize,
    pub deadline: usize,
    pub machine: MId,

}
impl Operation{
    pub fn new(name:String,id:OpId) -> Self{
        Operation{name: name,
                id: id,
                release: usize::MIN,
                processing: usize::MIN,
                deadline: usize::MAX,
                machine: MId::max()}
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine{
    name: String,
    m_id: MId
}
impl Machine{
    pub fn new(name:String,id:usize) -> Self{
        Machine{name:name,m_id:MId::new(id)}
    }
}

#[derive(Clone, Default, Eq)]
pub struct Instance {
    pub njobs : u16,
    pub nops :usize,
    pub nmachs : usize,
    pub ops: HashMap<OpId, Rc<Operation>>,
    pub machs: HashMap<MId, Rc<Machine>>,
    pub constraints: HashMap<OpId,Vec<Constraint>>
}

impl PartialEq for Instance{
    // Required method
    fn eq(&self, other: &Self) -> bool{
        true
    }

    // Provided method
    fn ne(&self, other: &Self) -> bool {
        true
     }
}

// pub struct Cg<'a>{
//     nodes: Vec<CgNode<'a>>,
//     edges: Vec<CgEdge<'a>>
// }

// pub struct CgNode<'a>{
//     id: u8,
//     op: &'a Operation,
//     incoming: Vec<&'a CgEdge<'a>>,
//     outgoing: Vec<&'a CgEdge<'a>>
// }

// pub struct CgEdge<'a>{
//     src: &'a CgNode<'a>,
//     dst: &'a CgNode<'a>,
//     weight: i32
// }