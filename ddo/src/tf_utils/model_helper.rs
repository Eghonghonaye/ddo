/// example based on https://stackoverflow.com/a/68567498

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor, TensorType};
use crate::Decision;
use std::path::Path;

pub struct Model{
    graph: Graph,
    bundle: SavedModelBundle,
    input_name: String,
    output_name: String,
}

pub trait ModelHelper{

    type State;
    type DecisionState;
    // type OutputTensor where Self::OutputTensor: Copy;
    type OutputTensor;

    //TODO who creates the initial model
    fn load_model<P: AsRef<Path>>(model_path: P, model: &mut Model) {
        // Initialize an empty graph
        let mut graph = Graph::new();
        // Load saved model bundle (session state + meta_graph data)
        let bundle = 
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_path)
            .expect("Can't load saved model");

        model.graph = graph;
        model.bundle = bundle;
    }

    fn state_to_input_tensor(&self,state: Self::State) -> Tensor<f32>;

    fn perform_inference(&self, model:Model, state: Self::State) -> <Self as ModelHelper>::OutputTensor
    where <Self as ModelHelper>::OutputTensor: TensorType + Copy {
        let signature_input_parameter_name = model.input_name;
        let signature_output_parameter_name = model.output_name;

        // initialise inout tensor
        let tensor: Tensor<f32> = self.state_to_input_tensor(state);

        // Get the session from the loaded model bundle
        let session = &model.bundle.session;

        // Get signature metadata from the model bundle
        //TO DO: Unclear what reusing a session over and over does here tbh -  should we somehow "clear" and restart it?
        // Do any states in the model change due to inference? - PS: It shouldnt!
        let signature = model.bundle
            .meta_graph_def()
            .get_signature("serving_default")
            .unwrap();

        // Get input/output info
        let input_info = signature.get_input(&signature_input_parameter_name).unwrap();
        let output_info = signature.get_output(&signature_output_parameter_name).unwrap();

        // Get input/output ops from graph
        let input_op = model.graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        let output_op = model.graph
            .operation_by_name_required(&output_info.name().name)
            .unwrap();
        
        // Manages inputs and outputs for the execution of the graph
        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op, 0, &tensor); // Add any inputs

        let out = args.request_fetch(&output_op, 0); // Request outputs

        // Run model
        session.run(&mut args) // Pass to session to run
            .expect("Error occurred during calculations");

        // Fetch outputs after graph execution
        let out_res: Self::OutputTensor = args.fetch(out).unwrap()[0];

        return out_res;
    }

    fn extract_decision_from_model_output(&self, output: Self::OutputTensor) -> Decision<Self::DecisionState>
    where <Self as ModelHelper>::OutputTensor: Copy;
}
