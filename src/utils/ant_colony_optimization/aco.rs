use std::collections::HashSet;

use rand::Rng;

use super::aco_parameters::ACOParameters;
use crate::utils::data_handling::named_values_collection::SetOperationsTrait;

pub trait ACO {
    fn set_initial_pheromone_trails(&mut self, p: &ACOParameters);

    fn tau_factor_of_vertex(&self, vertex: &usize, current_clique: &HashSet<usize>) -> f64;

    fn increment_pheromone(
        &mut self,
        p: &ACOParameters,
        pheromone_delta: &f64,
        current_clique: &HashSet<usize>,
    );

    fn decrement_pheromone(&mut self, p: &ACOParameters);

    fn update_pheromone_trail(
        &mut self,
        p: &ACOParameters,
        global_best_clique: &HashSet<usize>,
        k_best_clique: &HashSet<usize>,
    ) {
        self.decrement_pheromone(p);
        let pheromone_delta =
            1.0 / (1.0 + global_best_clique.len() as f64 - k_best_clique.len() as f64);
        self.increment_pheromone(p, &pheromone_delta, k_best_clique);
    }

    fn vertex_probability(
        &self,
        p: &ACOParameters,
        vertex: &usize,
        current_clique: &HashSet<usize>,
        memory_tau: &mut Vec<f64>,
        memory_sum_tau_candidates: &f64,
    ) -> f64 {
        let tau: f64 = if memory_tau[*vertex] > 0.0 {
            memory_tau[*vertex]
        } else {
            self.tau_factor_of_vertex(vertex, current_clique)
        };

        let tau = tau.powf(p.alpha);
        tau / memory_sum_tau_candidates
    }

    fn choose_vertex_using_pheromones_probabilities(
        &self,
        p: &mut ACOParameters,
        candidates: &HashSet<usize>,
        current_clique: &HashSet<usize>,
    ) -> usize {
        let mut probabilities: Vec<(usize, f64)> = vec![(0, 0.0); candidates.len()];

        let mut memory_tau = vec![-1.0; p.graph.n_vertex];
        let mut memory_sum_tau_candidates = 0.0;
        for other_v in candidates.iter() {
            let tau_other_v: f64 = if memory_tau[*other_v] > 0.0 {
                memory_tau[*other_v]
            } else {
                self.tau_factor_of_vertex(other_v, current_clique)
            };

            memory_tau[*other_v] = tau_other_v;
            memory_sum_tau_candidates += tau_other_v;
        }

        let mut sum_propapilites = 0.0;
        for (i, candidate) in candidates.iter().enumerate() {
            let p = self.vertex_probability(
                p,
                candidate,
                current_clique,
                &mut memory_tau,
                &memory_sum_tau_candidates,
            );
            sum_propapilites += p;
            probabilities[i] = (*candidate, sum_propapilites);
        }

        let random: f64 = p.rand.gen_range(0.0..1.0);
        // println!("{}", random);
        for prob in &probabilities {
            if random <= prob.1 {
                return prob.0;
            }
        }

        probabilities.last().unwrap().0
    }

    fn aco_procedure(&mut self, p: &mut ACOParameters) -> HashSet<usize> {
        let mut global_best: HashSet<usize> = HashSet::new();

        for _gen in 0..p.cycles {
            let mut gen_best: HashSet<usize> = HashSet::new();

            for _k in 0..p.ants {
                let initial_vertex = p.graph.select_random_vertex();
                let mut k_clique: HashSet<usize> = HashSet::from_iter(vec![initial_vertex]);
                let mut candidates = p.graph.get_neighbor_candidates(initial_vertex);

                while !candidates.is_empty() {
                    let new_v = self.choose_vertex_using_pheromones_probabilities(
                        p,
                        &candidates,
                        &k_clique,
                    );
                    let new_v_is_semantically_valid =
                        self.candidate_is_semantically_valid(p, &new_v, &k_clique);
                    if new_v_is_semantically_valid {
                        // println!("new_v: {}, current_clique: {:?}", new_v, &k_clique);
                        let new_v_candidates = p.graph.get_neighbor_candidates(new_v);
                        k_clique.insert(new_v);
                        candidates = candidates
                            .intersection(&new_v_candidates)
                            .copied()
                            .collect();
                    } else {
                        candidates.remove(&new_v);
                    }
                }

                gen_best = Self::choose_best_clique(p, &gen_best, &k_clique);
            }

            global_best = Self::choose_best_clique(p, &global_best, &gen_best);
            self.update_pheromone_trail(p, &global_best, &gen_best);
        }

        global_best
    }

    fn candidate_is_semantically_valid(
        &self,
        p: &ACOParameters,
        candidate: &usize,
        current_clique: &HashSet<usize>,
    ) -> bool {
        let mut is_valid: bool = true;

        // check if new cliques clause is complete
        let mut new_clique = current_clique.clone();
        new_clique.insert(*candidate);
        let new_clique_clause = p.graph.get_clique_clause(&new_clique);

        // loop every positive element
        for positive in &p.graph.positive_dataset {
            let mut is_valid_one_positive = false;
            let intersect = new_clique_clause.intersection(&positive.attributes);
            for (_, (_, i)) in intersect {
                is_valid_one_positive = is_valid_one_positive || !i.is_empty();
            }

            is_valid = is_valid && is_valid_one_positive;
        }

        is_valid
    }

    fn choose_best_clique(
        p: &mut ACOParameters,
        clique_1: &HashSet<usize>,
        clique_2: &HashSet<usize>,
    ) -> HashSet<usize> {
        match clique_1.len().cmp(&clique_2.len()) {
            std::cmp::Ordering::Greater => clique_1.clone(),
            std::cmp::Ordering::Less => clique_2.clone(),
            std::cmp::Ordering::Equal => {
                let random = p.rand.gen_ratio(1, 2);
                if random {
                    clique_1.clone()
                } else {
                    clique_2.clone()
                }
            }
        }
    }
}
