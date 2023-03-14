use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use utils::{
    ant_colony_optimization::edge_ac::EdgeAC,
    data_handling::{dataset::Dataset, named_values_set_list::SetOperationsTrait},
    graph::rejectability::create_rejectability_graph,
};

use crate::utils::{
    ant_colony_optimization::{aco::ACO, aco_parameters::ACOParameters, vertex_ac::VertexAC},
    clause::{cnf::Cnf, disjunctive_clause::DisjunctiveClause},
    config::aco_config::ACOConfig,
};
pub mod utils;

#[pyfunction]
fn oqat_with_aco(
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    learning_class: f64,
    column_names: Vec<String>,
    _column_types: Vec<String>,
    aco_config: ACOConfig,
) -> PyResult<(
    Vec<Vec<(String, String, f64)>>,
    Vec<usize>,
    Vec<(String, Vec<f64>)>,
    Vec<(String, Vec<f64>)>,
)> {
    let dataset = Dataset::from_data(x_train, y_train, column_names, OrderedFloat(learning_class));

    let rng = StdRng::seed_from_u64(42);
    let (graph, most_representative, least_representative) =
        create_rejectability_graph(rng, dataset);

    let rng = StdRng::seed_from_u64(42);
    let mut aco_parameters = ACOParameters {
        graph,
        rand: rng,
        cycles: aco_config.cycles,
        ants: aco_config.ants,
        alpha: aco_config.alpha,
        rho: aco_config.rho,
        tau_max: aco_config.tau_max,
        tau_min: aco_config.tau_min,
    };

    let mut clique_sizes: Vec<usize> = vec![];
    let model = match aco_config.algorithm.as_str() {
        "vertex-ac" => {
            let mut vertex_ac = VertexAC::new(&aco_parameters);
            let mut cnf = Cnf::new();

            while !aco_parameters.graph.available_vertex.is_empty() {
                let best_clique = vertex_ac.aco_procedure(&mut aco_parameters);
                clique_sizes.push(best_clique.len());
                aco_parameters
                    .graph
                    .remove_vertex_set_from_available(&best_clique);

                let clause = aco_parameters.graph.get_clique_clause(best_clique);

                let clause = DisjunctiveClause::from_named_values_set_list(&clause);

                cnf.clauses.push(clause);
            }

            cnf.to_export_format()
        }
        "edge-ac" => {
            let mut vertex_ac = EdgeAC::new(&aco_parameters);
            let mut cnf = Cnf::new();

            while !aco_parameters.graph.available_vertex.is_empty() {
                let best_clique = vertex_ac.aco_procedure(&mut aco_parameters);
                clique_sizes.push(best_clique.len());
                aco_parameters
                    .graph
                    .remove_vertex_set_from_available(&best_clique);

                let clause = aco_parameters.graph.get_clique_clause(best_clique);

                let clause = DisjunctiveClause::from_named_values_set_list(&clause);

                cnf.clauses.push(clause);
            }

            cnf.to_export_format()
        }
        _ => vec![vec![]],
    };

    Ok((
        model,
        clique_sizes,
        most_representative.to_export_format(),
        least_representative.to_export_format(),
    ))
}

/// A Python module implemented in Rust.
#[pymodule]
fn simple_oqat(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(oqat_with_aco, m)?)?;
    Ok(())
}
