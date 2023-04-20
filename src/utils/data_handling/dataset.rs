use std::fmt;

use ordered_float::OrderedFloat;

use crate::utils::data_handling::named_values_collection::{
    NamedValuesCollection, SetOperationsTrait,
};

use super::row::Row;

#[derive(Debug)]
pub struct Dataset {
    pub learning_pos: Vec<Row>,
    pub learning_neg: Vec<Row>,
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Learning set:")?;
        for (idx, row) in self.learning_pos.iter().enumerate() {
            writeln!(f, "  +{} {}", idx, row)?;
        }
        for (idx, row) in self.learning_neg.iter().enumerate() {
            writeln!(f, "  -{} {}", idx, row)?;
        }
        write!(f, "")
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset {
    pub fn new() -> Dataset {
        Dataset {
            learning_pos: Vec::new(),
            learning_neg: Vec::new(),
        }
    }

    pub fn from_data(
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>,
        column_names: Vec<String>,
        column_data_types: Vec<String>,
        learning_class: OrderedFloat<f64>,
    ) -> Dataset {
        let learning_rows =
            Dataset::create_rows_list(x_train, y_train, column_names, column_data_types);

        // Divide data into positive and negative elements
        let learning_pos: Vec<Row> = learning_rows
            .iter()
            .filter(|row| row.class == learning_class)
            .cloned()
            .collect();
        let learning_neg: Vec<Row> = learning_rows
            .iter()
            .filter(|row| row.class != learning_class)
            .cloned()
            .collect();

        Dataset {
            learning_pos,
            learning_neg,
        }
    }

    fn create_rows_list(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        column_names: Vec<String>,
        column_data_types: Vec<String>,
    ) -> Vec<Row> {
        assert!(x.len() == y.len());
        let mut rows = Vec::new();
        for (idx, x_values) in x.iter().enumerate() {
            let values: Vec<Vec<f64>> = x_values
                .iter()
                .map(|x| if *x >= 0. { vec![*x] } else { vec![] })
                .collect();
            let row = Row {
                class: OrderedFloat::from(y[idx]),
                attributes: NamedValuesCollection::from_f64_vec(
                    &column_names,
                    &column_data_types,
                    &values,
                ),
            };

            rows.push(row);
        }

        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_rows_list() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y = vec![1.0, 2.0];
        let column_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let column_data_types = vec!["num".to_string(), "num".to_string(), "cat".to_string()];
        let rows = Dataset::create_rows_list(x, y, column_names, column_data_types);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].class, OrderedFloat(1.0));
        assert_eq!(rows[1].class, OrderedFloat(2.0));
        assert_eq!(rows[0].attributes.len(), 3);
        assert_eq!(rows[1].attributes.len(), 3);
        let a1 = rows[0].attributes.get("a").unwrap();
        assert_eq!(a1.0, "num");
        assert_eq!(a1.1.len(), 1);
        assert!(a1.1.contains(&OrderedFloat(1.0)));

        let b1 = rows[0].attributes.get("b").unwrap();
        assert_eq!(b1.0, "num");
        assert_eq!(b1.1.len(), 1);
        assert!(b1.1.contains(&OrderedFloat(2.0)));

        let c1 = rows[0].attributes.get("c").unwrap();
        assert_eq!(c1.0, "cat");
        assert_eq!(c1.1.len(), 1);
        assert!(c1.1.contains(&OrderedFloat(3.0)));

        let a2 = rows[1].attributes.get("a").unwrap();
        assert_eq!(a2.0, "num");
        assert_eq!(a2.1.len(), 1);
        assert!(a2.1.contains(&OrderedFloat(4.0)));

        let b2 = rows[1].attributes.get("b").unwrap();
        assert_eq!(b2.0, "num");
        assert_eq!(b2.1.len(), 1);
        assert!(b2.1.contains(&OrderedFloat(5.0)));

        let c2 = rows[1].attributes.get("c").unwrap();
        assert_eq!(c2.0, "cat");
        assert_eq!(c2.1.len(), 1);
        assert!(c2.1.contains(&OrderedFloat(6.0)));
    }

    #[test]
    fn test_create_rows_list_with_missing_values() {
        let x = vec![vec![1.0, -1., 3.0], vec![4.0, -1., -1.]];
        let y = vec![1.0, 2.0];
        let column_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let column_data_types = vec!["num".to_string(), "num".to_string(), "cat".to_string()];
        let rows = Dataset::create_rows_list(x, y, column_names, column_data_types);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].class, OrderedFloat(1.0));
        assert_eq!(rows[1].class, OrderedFloat(2.0));
        assert_eq!(rows[0].attributes.len(), 3);
        assert_eq!(rows[1].attributes.len(), 3);

        let a1 = rows[0].attributes.get("a").unwrap();
        assert_eq!(a1.0, "num");
        assert_eq!(a1.1.len(), 1);
        assert!(a1.1.contains(&OrderedFloat(1.0)));

        let b1 = rows[0].attributes.get("b").unwrap();
        assert_eq!(b1.0, "num");
        assert_eq!(b1.1.len(), 0);

        let c1 = rows[0].attributes.get("c").unwrap();
        assert_eq!(c1.0, "cat");
        assert_eq!(c1.1.len(), 1);
        assert!(c1.1.contains(&OrderedFloat(3.0)));

        let a2 = rows[1].attributes.get("a").unwrap();
        assert_eq!(a2.0, "num");
        assert_eq!(a2.1.len(), 1);
        assert!(a2.1.contains(&OrderedFloat(4.0)));

        let b2 = rows[1].attributes.get("b").unwrap();
        assert_eq!(b2.0, "num");
        assert_eq!(b2.1.len(), 0);

        let c2 = rows[1].attributes.get("c").unwrap();
        assert_eq!(c2.0, "cat");
        assert_eq!(c2.1.len(), 0);
    }

    #[test]
    fn create_dataset_from_data() {
        let x_train = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y_train = vec![1.0, 2.0];
        let column_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let column_data_types = vec!["num".to_string(), "num".to_string(), "cat".to_string()];

        let dataset = Dataset::from_data(
            x_train,
            y_train,
            column_names,
            column_data_types,
            OrderedFloat(1.0),
        );
        assert_eq!(dataset.learning_pos.len(), 1);
        assert_eq!(dataset.learning_neg.len(), 1);
        assert_eq!(dataset.learning_pos[0].class, OrderedFloat(1.0));
        assert_eq!(dataset.learning_neg[0].class, OrderedFloat(2.0));
    }
}
