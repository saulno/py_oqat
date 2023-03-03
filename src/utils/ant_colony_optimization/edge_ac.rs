use std::collections::HashSet;

use super::{aco::ACO, aco_parameters::ACOParameters};

pub struct EdgeAC {
    pheromones: Vec<Vec<f64>>,
}

impl EdgeAC {
    pub fn new(parameters: &ACOParameters) -> EdgeAC {
        let mut e = EdgeAC {
            pheromones: vec![vec![]],
        };
        e.set_initial_pheromone_trails(parameters);

        e
    }
}

impl ACO for EdgeAC {
    fn set_initial_pheromone_trails(&mut self, p: &ACOParameters) {
        self.pheromones = vec![vec![p.tau_max; p.graph.n_vertex]; p.graph.n_vertex];
    }

    fn tau_factor_of_vertex(&self, vertex: &usize, current_clique: &HashSet<usize>) -> f64 {
        let mut sum: f64 = 0.0;
        for v in current_clique {
            sum += self.pheromones[*vertex][*v]
        }

        sum
    }

    fn increment_pheromone(
        &mut self,
        p: &ACOParameters,
        pheromone_delta: &f64,
        current_clique: &HashSet<usize>,
    ) {
        for i in 0..current_clique.len() {
            for j in 0..current_clique.len() {
                if i != j {
                    self.pheromones[i][j] += pheromone_delta;
                    if self.pheromones[i][j] > p.tau_max {
                        self.pheromones[i][j] = p.tau_max;
                    }
                }
            }
        }
    }

    fn decrement_pheromone(&mut self, p: &ACOParameters) {
        for i in 0..self.pheromones.len() {
            for j in 0..self.pheromones[i].len() {
                self.pheromones[i][j] *= p.rho;
                if self.pheromones[i][j] < p.tau_min {
                    self.pheromones[i][j] = p.tau_min;
                }
            }
        }
    }
}
