use std::collections::HashSet;

use super::{aco::ACO, aco_parameters::ACOParameters};

pub struct VertexAC {
    pheromones: Vec<f64>,
}

impl VertexAC {
    pub fn new(parameters: &ACOParameters) -> VertexAC {
        let mut v = VertexAC { pheromones: vec![] };
        v.set_initial_pheromone_trails(parameters);

        v
    }
}

impl ACO for VertexAC {
    fn set_initial_pheromone_trails(&mut self, p: &ACOParameters) {
        self.pheromones = vec![p.tau_max; p.graph.n_vertex];
    }

    fn tau_factor_of_vertex(&self, vertex: &usize, _current_clique: &HashSet<usize>) -> f64 {
        self.pheromones[*vertex]
    }

    fn increment_pheromone(
        &mut self,
        p: &ACOParameters,
        pheromone_delta: &f64,
        _current_clique: &HashSet<usize>,
    ) {
        for i in 0..self.pheromones.len() {
            self.pheromones[i] += pheromone_delta;
            if self.pheromones[i] > p.tau_max {
                self.pheromones[i] = p.tau_max;
            }
        }
    }

    fn decrement_pheromone(&mut self, p: &ACOParameters) {
        for i in 0..self.pheromones.len() {
            self.pheromones[i] *= p.rho;
            if self.pheromones[i] < p.tau_min {
                self.pheromones[i] = p.tau_min;
            }
        }
    }
}
