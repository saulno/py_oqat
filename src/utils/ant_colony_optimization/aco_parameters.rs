use rand::{rngs::StdRng, SeedableRng};

use crate::utils::graph::rejectability_graph::Graph;

#[derive(Debug)]
pub struct ACOParameters {
    pub graph: Graph,
    pub rand: StdRng,
    pub cycles: usize,
    pub ants: usize,
    pub alpha: f64,
    pub rho: f64,
    pub tau_max: f64,
    pub tau_min: f64,
}

#[derive(Debug)]
pub enum ACOAlgorithm {
    VertexAC,
    EdgeAC,
}

impl ACOParameters {
    pub fn new(
        cycles: usize,
        ants: usize,
        alpha: f64,
        rho: f64,
        tau_max: f64,
        tau_min: f64,
    ) -> ACOParameters {
        let rng = StdRng::seed_from_u64(1000);
        ACOParameters {
            graph: Graph::new(rng.clone(), 0, vec![], vec![]),
            rand: rng,
            cycles,
            ants,
            alpha,
            rho,
            tau_max,
            tau_min,
        }
    }
}
