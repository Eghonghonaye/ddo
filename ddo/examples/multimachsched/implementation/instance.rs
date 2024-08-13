use crate::abstraction::instance::{Instance,Operation,Machine,OpId,MId};
use crate::abstraction::constraints::{Constraint,Satisfaction,Deadline,Release,Setup,SetupType,Processing,Precedence,Assign,NoRepeat};
use crate::bitvector::BitVector;
use crate::model::State;
use std::collections::HashMap;

use std::fs::File;
use std::path::Path;
use std::io::{BufRead, BufReader};

use std::rc::Rc;
use std::borrow::BorrowMut;

use regex::Regex;

impl Instance{
    fn new() -> Self {
        Instance {njobs : 0, 
                    nops : 0, 
                    nmachs : 0,
                    ops : HashMap::new(),
                    machs : HashMap::new(),
                    constraints:HashMap::new(),
                    }
    }

    fn add_op(&mut self,name:String,id:usize) -> (String,OpId) {
        self.nops += 1;
        // self.ops.insert(OpId::new(id),Rc::new(Operation::new(name.clone(),OpId::new(id))));
        self.ops.insert(OpId::new(id),Operation::new(name.clone(),OpId::new(id)));
        (name,OpId::new(id))
    }

    fn add_machine(&mut self,name:String,mid:usize) -> (String,MId) {
        self.nmachs += 1;
        self.machs.insert(MId::new(mid),Rc::new(Machine::new(name.clone(),mid)));
        (name,MId::new(mid))
    }

    fn add_constraint(&mut self,op:OpId,cons:Constraint){
        match self.constraints.get_mut(&op){
            Some(op_cons) => {op_cons.push(cons.clone());},
            None => {self.constraints.insert(op,vec![cons.clone()]);}
        }
        // self.constraints.push(cons.clone());
    }

    pub fn from_file<P>(filename: P) -> Self 
    where P: AsRef<Path>,{
        let operations_re   = Regex::new(r"(v:)\s+(o:)\s+(op).(?P<job_id>\d+),(?P<op_id>\d+)[^\d]*(;)$").unwrap();
        let machines_re    = Regex::new(r"(v:)\s+(m:)\s+(m).(?P<mid>\d+)[^\d]*(;)*$").unwrap();

        let assignments_re   = Regex::new(r"(c:)\s+(assign)[^\d]*(?P<job_id>\d+),(?P<op_id>\d+)[^\d]*(?P<mid>\d+)[^\d]*(;)*$").unwrap();
        let release_re    = Regex::new(r"(c:)\s+(release)[^\d]*(?P<job_id>\d+),(?P<op_id>\d+)[^\d]*(?P<val>\d+)[^\d]*(;)*$").unwrap();
        let precedence_re = Regex::new(r"(c:)\s+(precedence)[^\d]*(?P<job_id_a>\d+),(?P<op_id_a>\d+)[^\d]*(?P<job_id_b>\d+),(?P<op_id_b>\d+)[^\d]*(;)*$").unwrap();
        let setup_re = Regex::new(r"(c:)\s+(setup)[^\d]*(?P<job_id_a>\d+),(?P<op_id_a>\d+)[^\d]*(?P<job_id_b>\d+),(?P<op_id_b>\d+)[^\d]*(?P<val>\d+)[^\d]*(;)*$").unwrap();
        let deadline_re = Regex::new(r"(c:)\s+(deadline)[^\d]*(?P<job_id>\d+),(?P<op_id>\d+)[^\d]*(?P<val>\d+)[^\d]*(;)*$").unwrap();
        let processing_re = Regex::new(r"(c:)\s+(processing)[^\d]*(?P<job_id>\d+),(?P<op_id>\d+)[^\d]*(?P<val>\d+)[^\d]*(;)*$").unwrap();
        let norepeat_re = Regex::new(r"(c:)\s+(norepeat)[^\d]*(?P<job_id>\d+),(?P<op_id>\d+)[^\d]*(;)*$").unwrap();

        let mut instance = Instance::new();

        let file = File::open(filename).expect("could not open file");
        let mut op_name_to_id: HashMap<String, OpId> = HashMap::new();
        let mut mch_name_to_id: HashMap<String, MId> = HashMap::new();
        if let Ok::<std::io::Lines<BufReader<File>>, File>(lines) = Ok(BufReader::new(file).lines()){
            for line in lines {
                let line: String = line.unwrap();
                let line = line.trim();
                // println!("line is {}",line);
                if line.is_empty() {
                    continue;
                }

                // skip comments
                if line.starts_with("// ") {
                    continue;
                }
                if let Some(caps) = operations_re.captures(line) {
                    let job_id = caps["job_id"].to_string().parse::<String>().unwrap();
                    let op_id = caps["op_id"].to_string().parse::<String>().unwrap();
                    let op_name = format!("{job_id}_{op_id}");
                    let (name,id) = instance.add_op(op_name.clone(),instance.nops);
                    // TODO: make add opp return what it just added or directly populate this to prevent bugs
                    op_name_to_id.insert(name.clone(),id);
                    println!("found an op {},{}",job_id,op_id);
                    continue;
                }
                else if let Some(caps) = machines_re.captures(line) {
                    let mid = caps["mid"].to_string().parse::<usize>().unwrap();
                    let m_name = format!("{mid}");
                    let (name,mid) = instance.add_machine(m_name.clone(),instance.nmachs);
                    // TODO: make add machine return what it just added or directly populate this to prevent bugs
                    mch_name_to_id.insert(name.clone(),mid);
                    println!("found a machine {}",mid.as_string());
                    continue;
                }
                else if let Some(caps) = assignments_re.captures(line) {
                    let job_id = caps["job_id"].to_string().parse::<u8>().unwrap();
                    let op_id = caps["op_id"].to_string().parse::<u8>().unwrap();
                    let mid = caps["mid"].to_string().parse::<u8>().unwrap();
                    let op_name = format!("{job_id}_{op_id}");
                    let m_name = format!("{mid}");
                    if op_name_to_id.contains_key(&op_name) && mch_name_to_id.contains_key(&m_name){
                        let op_id = op_name_to_id.get(&op_name).unwrap();
                        let m_id = mch_name_to_id.get(&m_name).unwrap();
                        if instance.ops.contains_key(&op_id) &&instance.machs.contains_key(&m_id){
                            // instance.add_constraint(*op_id,Constraint::AssignCons(Assign{op_a: Rc::clone(instance.ops.get(&op_id).unwrap()),
                            //             mach: Rc::clone(instance.machs.get(&m_id).unwrap())})); 
                            instance.add_constraint(*op_id,Constraint::AssignCons(Assign{op_a: *op_id,
                                mach: Rc::clone(instance.machs.get(&m_id).unwrap())})); 
                                        println!("adding assignment constraint op {} to machine {}",op_id.as_string(),m_id.as_string());
                            if let Some(x) = instance.ops.get_mut(&op_id) { 
                                                                x.assign_machine(*m_id) };
                                                            
                            }
                        }
                    println!("found assignment of op {},{} to machine {}",job_id,op_id,mid);                        
                    continue;
                
                }
                else if let Some(caps) = release_re.captures(line) {
                    let job_id = caps["job_id"].to_string().parse::<u8>().unwrap();
                    let op_id = caps["op_id"].to_string().parse::<u8>().unwrap();
                    let val = caps["val"].to_string().parse::<usize>().unwrap();
                    let op_name = format!("{job_id}_{op_id}");
                    if op_name_to_id.contains_key(&op_name){
                        let op_id = op_name_to_id.get(&op_name).unwrap();
                        instance.add_constraint(*op_id,Constraint::ReleaseCons(Release{op_a: *op_id,
                                        value:val})); 
                        if let Some(x) = instance.ops.get_mut(&op_id) { 
                            x.set_release(val) };
                    }
                    println!("found release of op {},{} as {}",job_id,op_id,val);                        
                    continue;
                }
                else if let Some(caps) = setup_re.captures(line) {
                    let job_id_a = caps["job_id_a"].to_string().parse::<u8>().unwrap();
                    let op_id_a = caps["op_id_a"].to_string().parse::<u8>().unwrap();
                    let job_id_b = caps["job_id_b"].to_string().parse::<u8>().unwrap();
                    let op_id_b = caps["op_id_b"].to_string().parse::<u8>().unwrap();
                    let op_name_a = format!("{job_id_a}_{op_id_a}");
                    let op_name_b = format!("{job_id_b}_{op_id_b}");
                    let val = caps["val"].to_string().parse::<usize>().unwrap();
                    if op_name_to_id.contains_key(&op_name_a) && op_name_to_id.contains_key(&op_name_b){
                        let op_id_a = op_name_to_id.get(&op_name_a).unwrap();
                        let op_id_b = op_name_to_id.get(&op_name_b).unwrap();
                        instance.add_constraint(*op_id_b,Constraint::SetupCons(Setup{op_a: *op_id_a,
                                        op_b: *op_id_b,
                                        value: val,
                                        setup_type: SetupType::IndependentOps})); 
                    }
                    println!("found setup between of op {},{} op {},{} as {}",job_id_a,op_id_a,job_id_b,op_id_b,val);
                }
                else if let Some(caps) = precedence_re.captures(line) {
                    let job_id_a = caps["job_id_a"].to_string().parse::<u8>().unwrap();
                    let op_id_a = caps["op_id_a"].to_string().parse::<u8>().unwrap();
                    let job_id_b = caps["job_id_b"].to_string().parse::<u8>().unwrap();
                    let op_id_b = caps["op_id_b"].to_string().parse::<u8>().unwrap();
                    let op_name_a = format!("{job_id_a}_{op_id_a}");
                    let op_name_b = format!("{job_id_b}_{op_id_b}");
                    if op_name_to_id.contains_key(&op_name_a) && op_name_to_id.contains_key(&op_name_b){
                        let op_id_a = op_name_to_id.get(&op_name_a).unwrap();
                        let op_id_b = op_name_to_id.get(&op_name_b).unwrap();
                        instance.add_constraint(*op_id_b,Constraint::PrecedenceCons(Precedence{op_a: *op_id_a,
                            op_b: *op_id_b})); 
                    }
                    println!("found precedence between op {},{} op {},{}",job_id_a,op_id_a,job_id_b,op_id_b);
                }
                else if let Some(caps) = processing_re.captures(line) {
                    let job_id = caps["job_id"].to_string().parse::<u8>().unwrap();
                    let op_id = caps["op_id"].to_string().parse::<u8>().unwrap();
                    let val = caps["val"].to_string().parse::<usize>().unwrap();
                    let op_name = format!("{job_id}_{op_id}");
                    if op_name_to_id.contains_key(&op_name){
                        let op_id = op_name_to_id.get(&op_name).unwrap();
                        instance.add_constraint(*op_id, Constraint::ProcessingCons(Processing{op_a: *op_id,
                                        value: val})); 
                        if let Some(x) = instance.ops.get_mut(&op_id) { 
                            x.set_processing(val) };
                    }
                    println!("found processing of op {},{} as {}",job_id,op_id,val);  
                }
                else if let Some(caps) = deadline_re.captures(line) {
                    let job_id = caps["job_id"].to_string().parse::<u8>().unwrap();
                    let op_id = caps["op_id"].to_string().parse::<u8>().unwrap();
                    let val = caps["val"].to_string().parse::<usize>().unwrap();
                    let op_name = format!("{job_id}_{op_id}");
                    if op_name_to_id.contains_key(&op_name){
                        let op_id = op_name_to_id.get(&op_name).unwrap();
                        instance.add_constraint(*op_id,Constraint::DeadlineCons(Deadline{op_a: *op_id,
                                        value: val})); 
                        if let Some(x) = instance.ops.get_mut(&op_id) { 
                            x.set_deadline(val) };
                    }
                    println!("found deadline of op {},{} as {}",job_id,op_id,val);  
                }
                else if let Some(caps) = norepeat_re.captures(line) {
                    let job_id = caps["job_id"].to_string().parse::<u8>().unwrap();
                    let op_id = caps["op_id"].to_string().parse::<u8>().unwrap();
                    let op_name = format!("{job_id}_{op_id}");
                    if op_name_to_id.contains_key(&op_name){
                        let op_id = op_name_to_id.get(&op_name).unwrap();
                        instance.add_constraint(*op_id, Constraint::NoRepeatCons(NoRepeat{op_a: *op_id})); 
                    }
                    println!("found no repeat of op {},{}",job_id,op_id);  
                }
                else{
                    ()
                }
            }
        }
    println!("Instance has {} ops and {} machines and {} constraints",instance.nops,instance.nmachs,instance.constraints.len());
    for (opid,op) in &instance.ops{    println!("{:?},{:?}",opid,op.machine);  }
    instance
    }
    
    pub fn construct_feasible_set(&self, state: &State) -> BitVector{
        let mut feasible_set = BitVector::ones(self.nops); // initialise feasible set as all operations in the system
        for (affected_op,constraints) in &self.constraints{
            // delete any operations with a violated constraint
            // already scheduled, precedence not in some etc
            for constraint in constraints{
                constraint.filter_set(self,state,affected_op,&mut feasible_set);
            }
            
        }
        feasible_set
    }
}