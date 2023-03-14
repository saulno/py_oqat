use std::{collections::HashSet, fmt};

use ordered_float::OrderedFloat;

use crate::utils::data_handling::named_values_set::NamedValuesSet;

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
        learning_class: OrderedFloat<f64>,
    ) -> Dataset {
        let learning_rows = Dataset::create_rows_list(x_train, y_train, column_names);

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

    fn create_rows_list(x: Vec<Vec<f64>>, y: Vec<f64>, column_names: Vec<String>) -> Vec<Row> {
        assert!(x.len() == y.len());
        let mut rows = Vec::new();
        for (idx, x_values) in x.iter().enumerate() {
            let mut row = Row {
                class: OrderedFloat::from(y[idx]),
                attributes: Vec::new(),
            };
            for (idx, x_value) in x_values.iter().enumerate() {
                let mut values = HashSet::new();
                values.insert(OrderedFloat(*x_value));
                let named_values_set = NamedValuesSet {
                    column_name: column_names[idx].clone(),
                    values,
                };
                row.attributes.push(named_values_set);
            }
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
        let rows = Dataset::create_rows_list(x, y, column_names);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].class, OrderedFloat(1.0));
        assert_eq!(rows[1].class, OrderedFloat(2.0));
        assert_eq!(rows[0].attributes.len(), 3);
        assert_eq!(rows[1].attributes.len(), 3);
        assert_eq!(rows[0].attributes[0].column_name, "a");
        assert_eq!(rows[0].attributes[1].column_name, "b");
        assert_eq!(rows[0].attributes[2].column_name, "c");
        assert_eq!(rows[1].attributes[0].column_name, "a");
        assert_eq!(rows[1].attributes[1].column_name, "b");
        assert_eq!(rows[1].attributes[2].column_name, "c");
        assert_eq!(rows[0].attributes[0].values.len(), 1);
        assert_eq!(rows[0].attributes[1].values.len(), 1);
        assert_eq!(rows[0].attributes[2].values.len(), 1);
        assert_eq!(rows[1].attributes[0].values.len(), 1);
        assert_eq!(rows[1].attributes[1].values.len(), 1);
        assert_eq!(rows[1].attributes[2].values.len(), 1);
        assert_eq!(
            rows[0].attributes[0].values.contains(&OrderedFloat(1.0)),
            true
        );
        assert_eq!(
            rows[0].attributes[1].values.contains(&OrderedFloat(2.0)),
            true
        );
        assert_eq!(
            rows[0].attributes[2].values.contains(&OrderedFloat(3.0)),
            true
        );
        assert_eq!(
            rows[1].attributes[0].values.contains(&OrderedFloat(4.0)),
            true
        );
        assert_eq!(
            rows[1].attributes[1].values.contains(&OrderedFloat(5.0)),
            true
        );
        assert_eq!(
            rows[1].attributes[2].values.contains(&OrderedFloat(6.0)),
            true
        );
    }

    #[test]
    fn create_dataset_from_data() {
        let x_train = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y_train = vec![1.0, 2.0];
        let column_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let dataset = Dataset::from_data(x_train, y_train, column_names, OrderedFloat(1.0));
        assert_eq!(dataset.learning_pos.len(), 1);
        assert_eq!(dataset.learning_neg.len(), 1);
        assert_eq!(dataset.learning_pos[0].class, OrderedFloat(1.0));
        assert_eq!(dataset.learning_neg[0].class, OrderedFloat(2.0));
    }
}
