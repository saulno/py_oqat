use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use rand::{rngs::StdRng, Rng};

use crate::utils::data_handling::{
    named_values_collection::{NamedValuesCollection, SetOperationsTrait},
    row::Row,
};

use super::edge::Edge;

#[derive(Debug)]
pub struct Graph {
    pub adj_mtx: Vec<Vec<Edge>>,
    pub edge_dict: HashMap<usize, HashSet<usize>>,
    pub n_vertex: usize,
    pub n_edges: usize,
    pub available_vertex: HashSet<usize>,
    pub reject_one_negative: Vec<NamedValuesCollection>,
    pub positive_dataset: Vec<Row>,
    rng: StdRng,
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "Graph: ({} vertices, {} edges)",
            self.n_vertex, self.n_edges
        )?;
        for i in 0..self.n_vertex {
            for j in 0..self.n_vertex {
                if let Some(clause_values) = &self.adj_mtx[i][j] {
                    writeln!(f, "{} -> {} : {:?}\n", i, j, clause_values)?;
                }
            }
        }
        write!(f, "")
    }
}

impl Graph {
    pub fn new(
        rng: StdRng,
        num_vertex: usize,
        reject_one_negative: Vec<NamedValuesCollection>,
        positive_dataset: Vec<Row>,
    ) -> Graph {
        let mut graph = Graph {
            adj_mtx: vec![],
            edge_dict: HashMap::new(),
            n_vertex: 0,
            n_edges: 0,
            available_vertex: HashSet::new(),
            reject_one_negative,
            positive_dataset,
            rng,
        };

        graph.n_vertex = num_vertex;
        graph.adj_mtx = vec![vec![None; num_vertex]; num_vertex];

        for i in 0..num_vertex {
            graph.edge_dict.insert(i, HashSet::new());
            graph.available_vertex.insert(i);
        }

        graph
    }

    pub fn add_edge(&mut self, u: usize, v: usize, clause_values: &NamedValuesCollection) {
        self.adj_mtx[u][v] = Some(clause_values.clone());
        self.adj_mtx[v][u] = Some(clause_values.clone());

        self.edge_dict.get_mut(&u).unwrap().insert(v);
        self.edge_dict.get_mut(&v).unwrap().insert(u);

        if clause_values.len() > 0 {
            self.n_edges += 1;
        }
    }

    pub fn is_edge(&self, vertex_1: usize, vertex_2: usize) -> bool {
        self.adj_mtx[vertex_1][vertex_2].is_some()
    }

    pub fn select_random_vertex(&mut self) -> usize {
        loop {
            let selected = self.rng.gen_range(0..self.n_vertex);
            if self.available_vertex.contains(&selected) {
                return selected;
            }
        }
    }

    pub fn get_neighbor_candidates(&self, vertex: usize) -> HashSet<usize> {
        let all_neighbors = self.edge_dict.get(&vertex).unwrap();
        let available_neighbors = all_neighbors.intersection(&self.available_vertex);
        available_neighbors.copied().collect()
    }

    pub fn remove_vertex_set_from_available(&mut self, vertex_set: &HashSet<usize>) {
        self.available_vertex = self
            .available_vertex
            .difference(vertex_set)
            .copied()
            .collect();
    }

    pub fn get_clique_clause(&self, clique: &HashSet<usize>) -> NamedValuesCollection {
        let mut clique_clause = NamedValuesCollection::new();

        if clique.len() == 1 {
            let vertex = clique.iter().next().unwrap();
            return self.reject_one_negative[*vertex].clone();
        }

        let vertex_list = clique.iter().copied().collect::<Vec<usize>>();
        for i in 0..vertex_list.len() {
            for j in i + 1..vertex_list.len() {
                let (v1, v2): (usize, usize) = (vertex_list[i], vertex_list[j]);
                let edge_clause = self.adj_mtx[v1][v2].as_ref().unwrap();
                clique_clause = if clique_clause.is_empty() {
                    edge_clause.clone()
                } else {
                    clique_clause.intersection(edge_clause)
                };
            }
        }

        clique_clause
    }
}
