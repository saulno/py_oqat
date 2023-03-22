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
    dataset: &Dataset,
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

    // println!("complete clause {:?}", accept_all_positive);

    let mut graph = Graph::new(
        rng,
        dataset.learning_neg.len(),
        reject_only_one_negative,
        dataset.learning_pos.clone(),
    );

    // add an edge for every possible pair of negative examples
    // println!("Start graph construction");
    // println!(
    //     "Number of edges to verify {}",
    //     dataset.learning_neg.len() * (dataset.learning_neg.len() - 1) / 2
    // );
    // let mut edges_computed = 0;
    for i in 0..dataset.learning_neg.len() {
        for j in i + 1..dataset.learning_neg.len() {
            // get a list of sets with every selector of the two negative examples
            let negative_pair_attrs = construct_attribute_sets(&dataset.learning_neg, &[i, j]);
            // println!("negative pair attrs {}", negative_pair_attrs);

            // clause is a list of sets containing every selector
            // that is present in every positive element and
            // not in the two negative elements
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
                let singular_clause_two_neg_one_pos = find_clause_one_positive(
                    &dataset.learning_pos,
                    positive_idx,
                    &negative_pair_attrs,
                );

                // add to the clause, the new selectors for this positive element
                clause = clause.union(&singular_clause_two_neg_one_pos);
                // clause = update_clause(&clause, &singular_clause_two_neg_one_pos);

                exists_clause_for_all_positive =
                    exists_clause_for_all_positive && exists_clause_two_neg_on_pos;
            }

            if exists_clause_for_all_positive {
                graph.add_edge(i, j, &clause);
                // println!(
                //     "There's an edge between {} and {}, with clause {}",
                //     i, j, clause
                // );
            }

            // print progress
            // edges_computed += 1;
            // let quantile =
            //     (dataset.learning_neg.len() * (dataset.learning_neg.len() - 1) / 2) / 100;
            // if edges_computed % quantile == 0 {
            //     println!("{}% done", edges_computed / quantile);
            // }
        }
    }

    (graph, accept_all_positive, accept_all_negative)
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
        let (graph, _, _) = super::create_rejectability_graph(rng, &dataset);
        assert_eq!(graph.n_vertex, 21);
        assert_eq!(graph.n_edges, 174);
        assert_eq!(graph.edge_dict[&0].len(), 17);
        assert_eq!(graph.edge_dict[&1].len(), 17);
        assert_eq!(graph.edge_dict[&2].len(), 20);
        assert_eq!(graph.adj_mtx[0][1].as_ref().unwrap().len(), 3);
        // assert_eq!(graph.adj_mtx[0][1], None);
    }

    #[test]
    fn test_exist_clause_one_positive() {
        let dataset = create_mock_data();

        let positive_idx = 0;
        let (neg_1, neg_2) = (0, 1);
        let negative_pair_attrs =
        super::construct_attribute_sets(&dataset.learning_neg, &[neg_1, neg_2]);
        let exists_clause = super::exists_clause_one_positive(
            &dataset.learning_pos[positive_idx],
            &negative_pair_attrs,
        );
        
        assert!(exists_clause);

        let (neg_1, neg_2) = (0, 9);
        let negative_pair_attrs =
            super::construct_attribute_sets(&dataset.learning_neg, &[neg_1, neg_2]);
        let exists_clause = super::exists_clause_one_positive(
            &dataset.learning_pos[positive_idx],
            &negative_pair_attrs,
        );

        assert!(!exists_clause);
    }

    #[test]
    fn test_find_clause_one_positive() {
        let dataset = create_mock_data();

        let positive_idx = 0;
        let (neg_1, neg_2) = (0, 1);
        let negative_pair_attrs =
            super::construct_attribute_sets(&dataset.learning_neg, &[neg_1, neg_2]);
        let clause = super::find_clause_one_positive(
            &dataset.learning_pos,
            positive_idx,
            &negative_pair_attrs,
        );

        assert_eq!(clause.len(), 3);
        let a = &clause[0];
        assert_eq!(a.values.len(), 1);
        assert!(a.values.contains(&OrderedFloat(1.0)));
        let b = &clause[1];
        assert_eq!(b.values.len(), 0);
        let c = &clause[2];
        assert_eq!(c.values.len(), 0);
    }

    #[test]
    fn test_construct_attribute_sets() {
        let dataset = create_mock_data();
        let subset = vec![0, 1];

        let values = super::construct_attribute_sets(&dataset.learning_neg, &subset);

        assert_eq!(values.len(), 3);
        let a = &values[0];
        assert_eq!(a.values.len(), 1);
        assert!(a.values.contains(&OrderedFloat(0.0)));

        let b = &values[1];
        assert_eq!(b.values.len(), 1);
        assert!(b.values.contains(&OrderedFloat(2.0)));

        let c = &values[2];
        assert_eq!(c.values.len(), 2);
        assert!(c.values.contains(&OrderedFloat(0.0)));
        assert!(c.values.contains(&OrderedFloat(1.0)));
    }
}
