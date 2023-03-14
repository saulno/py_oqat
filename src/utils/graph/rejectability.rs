use std::{
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use cute::c;
use rand::rngs::StdRng;

use crate::utils::data_handling::{
    named_values_set::NamedValuesSet,
    named_values_set_list::{NamedValuesSetList, SetOperationsTrait},
};

use super::{
    super::data_handling::{dataset::Dataset, row::Row},
    rejectability_graph::Graph,
};

// create rejectability graph
pub fn create_rejectability_graph(
    rng: StdRng,
    dataset: Dataset,
) -> (Graph, NamedValuesSetList, NamedValuesSetList) {
    // create a complete clause (accepts all posotive)
    let accept_all_positive = construct_attribute_sets(
        &dataset.learning_pos,
        &c![i, for i in 0..dataset.learning_pos.len()],
    );
    let accept_all_negative = construct_attribute_sets(
        &dataset.learning_neg,
        &c![i, for i in 0..dataset.learning_neg.len()],
    );

    // for every negative element, create a clause that rejects only that element
    let mut reject_only_one_negative: Vec<NamedValuesSetList> = vec![];
    for neg in &dataset.learning_neg {
        let mut one_neg_clause = accept_all_positive.clone();
        one_neg_clause = one_neg_clause.difference(&neg.attributes);
        reject_only_one_negative.push(one_neg_clause);
    }

    let mut graph = Graph::new(
        rng,
        dataset.learning_neg.len(),
        reject_only_one_negative,
        dataset.learning_pos.clone(),
    );

    let edges_mutex: Arc<Mutex<Vec<(usize, usize, NamedValuesSetList)>>> =
        Arc::new(Mutex::new(vec![]));
    let mut threads: Vec<JoinHandle<()>> = vec![];

    for i in 0..dataset.learning_neg.len() {

        let ds = dataset.clone();
        let edges_mutex = Arc::clone(&edges_mutex);
        
        let thread = std::thread::spawn(move || {
            for j in i + 1..ds.learning_neg.len() {
                let mut edges = edges_mutex.lock().unwrap();
                let edge = find_edge_between(i, j, &ds);
                if let Some(edge) = edge {
                    edges.push((i, j, edge));
                }
            }
        });
        threads.push(thread);
    }
    for handle in threads {
        handle.join().unwrap();
    }

    let edges = edges_mutex.lock().unwrap();
    for (i, j, edge) in edges.iter() {
        graph.add_edge(*i, *j, edge);
    }

    (graph, accept_all_positive, accept_all_negative)
}

fn find_edge_between(i: usize, j: usize, dataset: &Dataset) -> Option<NamedValuesSetList> {
    // get a list of sets with every selector of the two negative examples
    let negative_pair_attrs = construct_attribute_sets(&dataset.learning_neg, &[i, j]);

    let mut clause: NamedValuesSetList =
        c![NamedValuesSet::new(), for _i in 0..dataset.learning_pos[0].attributes.len()];

    let mut exists_clause_for_all_positive = true;

    // check every element in the positive dataset to see if theres a complete clause tha rejects the pair
    for (positive_idx, positive) in dataset.learning_pos.iter().enumerate() {
        let exists_clause_two_neg_on_pos =
            exists_clause_one_positive(positive, &negative_pair_attrs);

        if !exists_clause_two_neg_on_pos {
            exists_clause_for_all_positive = false;
            break;
        }

        // find the clause that rejects the pair and accepts current positive element
        let singular_clause_two_neg_one_pos =
            find_clause_one_positive(&dataset.learning_pos, positive_idx, &negative_pair_attrs);

        // add to the clause, the new selectors for this positive element
        clause = clause.union(&singular_clause_two_neg_one_pos);

        exists_clause_for_all_positive =
            exists_clause_for_all_positive && exists_clause_two_neg_on_pos;
    }

    if exists_clause_for_all_positive {
        return Some(clause);
    } else {
        return None;
    }
}

pub fn exists_clause_one_positive(
    positive: &Row,
    negative_pair_attrs: &NamedValuesSetList,
) -> bool {
    let mut exists_clause = false;

    for (pos_attr_idx, pos_attr_set) in positive.attributes.iter().enumerate() {
        let neg_attr_set = &negative_pair_attrs[pos_attr_idx];
        let first_val_positive = pos_attr_set.values.iter().next().unwrap();
        exists_clause = exists_clause || !neg_attr_set.values.contains(first_val_positive);
    }

    exists_clause
}

pub fn find_clause_one_positive(
    positive_dataset: &[Row],
    positive_idx: usize,
    negative_pair_attrs: &NamedValuesSetList,
) -> NamedValuesSetList {
    let positive_element_attrs = construct_attribute_sets(positive_dataset, &[positive_idx]);
    let mut clause: NamedValuesSetList =
        c![NamedValuesSet::new(), for _i in 0..positive_dataset[0].attributes.len()];

    for (pos_attr_idx, pos_attr_set) in positive_element_attrs.iter().enumerate() {
        let neg_attr_set = &negative_pair_attrs[pos_attr_idx];
        clause[pos_attr_idx] = NamedValuesSet {
            column_name: neg_attr_set.column_name.clone(),
            values: pos_attr_set
                .values
                .difference(&neg_attr_set.values)
                .cloned()
                .collect(),
        };
    }

    clause
}

// construct a list of sets containig the values of every atrribute for each element in a subest of the dataset
pub fn construct_attribute_sets(dataset: &[Row], subset: &[usize]) -> NamedValuesSetList {
    let subset_elements = subset
        .iter()
        .map(|&i| dataset[i].attributes.clone())
        .collect::<Vec<_>>();

    let mut values = subset_elements[0].clone();

    for elem_set in subset_elements {
        values = values.union(&elem_set);
    }

    values
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use ordered_float::OrderedFloat;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::utils::data_handling::dataset::Dataset;

    fn create_mock_data() -> Dataset {
        let x_train = vec![
            vec![0.0, 2.0, 0.0],
            vec![0.0, 2.0, 1.0],
            vec![0.0, 2.0, 2.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 2.0],
            vec![1.0, 2.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![1.0, 2.0, 2.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 2.0],
            vec![2.0, 2.0, 0.0],
            vec![2.0, 2.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![2.0, 1.0, 0.0],
            vec![2.0, 1.0, 1.0],
            vec![2.0, 1.0, 2.0],
            vec![2.0, 0.0, 0.0],
            vec![2.0, 0.0, 1.0],
            vec![2.0, 0.0, 2.0],
        ];
        let y_train = vec![
            2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0,
        ];
        let column_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let dataset = Dataset::from_data(x_train, y_train, column_names, OrderedFloat(0.0));

        dataset
    }

    #[test]
    fn test_create_rejectability_graph() {
        let dataset = create_mock_data();
        let rng = StdRng::seed_from_u64(42);
        let (graph, _, _) = super::create_rejectability_graph(rng, dataset);
        assert_eq!(graph.n_vertex, 21);
        assert_eq!(
            graph.edge_dict[&0],
            HashSet::<usize>::from_iter(vec![
                1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20
            ])
        );
    }
}
