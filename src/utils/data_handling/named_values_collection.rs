use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;

pub type NamedValuesCollection = HashMap<String, (String, HashSet<OrderedFloat<f64>>)>;

pub trait SetOperationsTrait {
    fn new() -> Self;
    fn is_empty(&self) -> bool;
    fn from_f64_vec(
        column_names: &[String],
        column_data_types: &[String],
        values: &[Vec<f64>],
    ) -> Self;
    fn to_export_format(&self) -> Vec<(String, Vec<f64>)>;
    fn union(&self, other: &NamedValuesCollection) -> NamedValuesCollection;
    fn intersection(&self, other: &NamedValuesCollection) -> NamedValuesCollection;
    fn difference(&self, other: &NamedValuesCollection) -> NamedValuesCollection;
}

impl SetOperationsTrait for NamedValuesCollection {
    fn new() -> NamedValuesCollection {
        HashMap::new()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_f64_vec(
        column_names: &[String],
        column_data_types: &[String],
        values: &[Vec<f64>],
    ) -> NamedValuesCollection {
        let mut result = HashMap::new();
        for ((column_name, data_type), values) in column_names
            .iter()
            .zip(column_data_types.iter())
            .zip(values.iter())
        {
            result.insert(
                column_name.clone(),
                (
                    data_type.clone(),
                    values.iter().map(|x| OrderedFloat(*x)).collect(),
                ),
            );
        }

        result
    }

    fn to_export_format(&self) -> Vec<(String, Vec<f64>)> {
        self.iter()
            .map(|(column_name, (_data_type, values))| {
                (
                    column_name.clone(),
                    values.iter().map(|x| x.into_inner()).collect::<Vec<f64>>(),
                )
            })
            .collect::<Vec<(String, Vec<f64>)>>()
    }

    fn union(&self, other: &NamedValuesCollection) -> NamedValuesCollection {
        let mut result = HashMap::new();
        for (column_name, (data_type, current_values)) in self.iter() {
            if other.contains_key(column_name) {
                let (_, other_values) = &other[column_name];
                let unified_values: HashSet<OrderedFloat<f64>> =
                    current_values.union(other_values).cloned().collect();
                result.insert(column_name.clone(), (data_type.clone(), unified_values));
            } else {
                result.insert(
                    column_name.clone(),
                    (data_type.clone(), current_values.clone()),
                );
            }
        }
        for (column_name, (data_type, other_values)) in other.iter() {
            if !result.contains_key(column_name) {
                result.insert(
                    column_name.clone(),
                    (data_type.clone(), other_values.clone()),
                );
            }
        }

        result
    }

    fn intersection(&self, other: &NamedValuesCollection) -> NamedValuesCollection {
        let mut result = self.clone();
        for (column_name, (data_type, current_values)) in self.iter() {
            if other.contains_key(column_name) {
                let (_, other_values) = &other[column_name];
                let intersected_values: HashSet<OrderedFloat<f64>> =
                    current_values.intersection(other_values).cloned().collect();
                result.insert(column_name.clone(), (data_type.clone(), intersected_values));
            } else {
                result.insert(column_name.clone(), (data_type.clone(), HashSet::new()));
            }
        }
        for (column_name, (data_type, _current_values)) in other.iter() {
            if !result.contains_key(column_name) {
                result.insert(column_name.clone(), (data_type.clone(), HashSet::new()));
            }
        }

        result
    }

    fn difference(&self, other: &NamedValuesCollection) -> NamedValuesCollection {
        let mut result = self.clone();
        for (column_name, (data_type, current_values)) in self.iter() {
            if other.contains_key(column_name) {
                let (_, other_values) = &other[column_name];
                let subtracted_values: HashSet<OrderedFloat<f64>> =
                    current_values.difference(other_values).cloned().collect();
                result.insert(column_name.clone(), (data_type.clone(), subtracted_values));
            } else {
                result.insert(
                    column_name.clone(),
                    (data_type.clone(), current_values.clone()),
                );
            }
        }
        for (column_name, (data_type, _current_values)) in other.iter() {
            if !result.contains_key(column_name) {
                result.insert(column_name.clone(), (data_type.clone(), HashSet::new()));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union() {
        let mut list1 = NamedValuesCollection::new();
        list1.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![1.0, 2.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![3.0, 4.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![5.0, 6.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let mut list2 = NamedValuesCollection::new();
        list2.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![2.0, 3.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list2.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![4.0, 5.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list2.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![6.0, 7.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let unified = list1.union(&list2);
        assert_eq!(unified.len(), 3);
        let (_, a) = &unified["a"];
        assert_eq!(a.len(), 3);
        assert!(a.contains(&OrderedFloat(1.0)));
        assert!(a.contains(&OrderedFloat(2.0)));
        assert!(a.contains(&OrderedFloat(3.0)));
        let (_, b) = &unified["b"];
        assert_eq!(b.len(), 3);
        assert!(b.contains(&OrderedFloat(3.0)));
        assert!(b.contains(&OrderedFloat(4.0)));
        assert!(b.contains(&OrderedFloat(5.0)));
        let (_, c) = &unified["c"];
        assert_eq!(c.len(), 3);
        assert!(c.contains(&OrderedFloat(5.0)));
        assert!(c.contains(&OrderedFloat(6.0)));
        assert!(c.contains(&OrderedFloat(7.0)));

        let unified = list2.union(&list1);
        assert_eq!(unified.len(), 3);
        let (_, a) = &unified["a"];
        assert_eq!(a.len(), 3);
        assert!(a.contains(&OrderedFloat(1.0)));
        assert!(a.contains(&OrderedFloat(2.0)));
        assert!(a.contains(&OrderedFloat(3.0)));
        let (_, b) = &unified["b"];
        assert_eq!(b.len(), 3);
        assert!(b.contains(&OrderedFloat(3.0)));
        assert!(b.contains(&OrderedFloat(4.0)));
        assert!(b.contains(&OrderedFloat(5.0)));
        let (_, c) = &unified["c"];
        assert_eq!(c.len(), 3);
        assert!(c.contains(&OrderedFloat(5.0)));
        assert!(c.contains(&OrderedFloat(6.0)));
        assert!(c.contains(&OrderedFloat(7.0)));
    }

    #[test]
    fn test_union_with_empty_set() {
        let mut list1 = NamedValuesCollection::new();
        list1.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![1.0, 2.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![3.0, 4.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![5.0, 6.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let list2 = NamedValuesCollection::new();

        let unified = list1.union(&list2);
        assert_eq!(unified.len(), 3);
        let (_, a) = &unified["a"];
        assert_eq!(a.len(), 2);
        assert!(a.contains(&OrderedFloat(1.0)));
        assert!(a.contains(&OrderedFloat(2.0)));
        let (_, b) = &unified["b"];
        assert_eq!(b.len(), 2);
        assert!(b.contains(&OrderedFloat(3.0)));
        assert!(b.contains(&OrderedFloat(4.0)));
        let (_, c) = &unified["c"];
        assert_eq!(c.len(), 2);
        assert!(c.contains(&OrderedFloat(5.0)));
        assert!(c.contains(&OrderedFloat(6.0)));

        let unified = list2.union(&list1);
        assert_eq!(unified.len(), 3);
        let (_, a) = &unified["a"];
        assert_eq!(a.len(), 2);
        assert!(a.contains(&OrderedFloat(1.0)));
        assert!(a.contains(&OrderedFloat(2.0)));
        let (_, b) = &unified["b"];
        assert_eq!(b.len(), 2);
        assert!(b.contains(&OrderedFloat(3.0)));
        assert!(b.contains(&OrderedFloat(4.0)));
        let (_, c) = &unified["c"];
        assert_eq!(c.len(), 2);
        assert!(c.contains(&OrderedFloat(5.0)));
        assert!(c.contains(&OrderedFloat(6.0)));
    }

    #[test]
    fn test_intersection() {
        let mut list1 = NamedValuesCollection::new();
        list1.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![1.0, 2.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![3.0, 4.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![5.0, 6.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let mut list2 = NamedValuesCollection::new();
        list2.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![2.0, 3.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list2.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![4.0, 5.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list2.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![6.0, 7.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let intersected = list1.intersection(&list2);
        assert_eq!(intersected.len(), 3);
        let a = &intersected["a"];
        assert_eq!(a.1.len(), 1);
        assert!(a.1.contains(&OrderedFloat(2.0)));
        let b = &intersected["b"];
        assert_eq!(b.1.len(), 1);
        assert!(b.1.contains(&OrderedFloat(4.0)));
        let c = &&intersected["c"];
        assert_eq!(c.1.len(), 1);
        assert!(c.1.contains(&OrderedFloat(6.0)));

        let intersected = list2.intersection(&list1);
        assert_eq!(intersected.len(), 3);
        let a = &intersected["a"];
        assert_eq!(a.1.len(), 1);
        assert!(a.1.contains(&OrderedFloat(2.0)));
        let b = &intersected["b"];
        assert_eq!(b.1.len(), 1);
        assert!(b.1.contains(&OrderedFloat(4.0)));
        let c = &&intersected["c"];
        assert_eq!(c.1.len(), 1);
        assert!(c.1.contains(&OrderedFloat(6.0)));
    }

    #[test]
    fn test_intersection_with_empty_set() {
        let mut list1 = NamedValuesCollection::new();
        list1.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![1.0, 2.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![3.0, 4.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![5.0, 6.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let list2 = NamedValuesCollection::new();

        let intersected = list1.intersection(&list2);
        assert_eq!(intersected.len(), 3);
        let a = &intersected["a"];
        assert_eq!(a.1.len(), 0);
        let b = &intersected["b"];
        assert_eq!(b.1.len(), 0);
        let c = &&intersected["c"];
        assert_eq!(c.1.len(), 0);

        let intersected = list2.intersection(&list1);
        assert_eq!(intersected.len(), 3);
        let a = &intersected["a"];
        assert_eq!(a.1.len(), 0);
        let b = &intersected["b"];
        assert_eq!(b.1.len(), 0);
        let c = &&intersected["c"];
        assert_eq!(c.1.len(), 0);
    }

    #[test]
    fn test_difference() {
        let mut list1 = NamedValuesCollection::new();
        list1.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![1.0, 2.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![3.0, 4.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![5.0, 6.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let mut list2 = NamedValuesCollection::new();
        list2.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![2.0, 3.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list2.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![4.0, 5.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list2.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![6.0, 7.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let subtracted = list1.difference(&list2);
        assert_eq!(subtracted.len(), 3);
        let a = &subtracted["a"];
        assert_eq!(a.1.len(), 1);
        assert!(a.1.contains(&OrderedFloat(1.0)));
        let b = &subtracted["b"];
        assert_eq!(b.1.len(), 1);
        assert!(b.1.contains(&OrderedFloat(3.0)));
        let c = &subtracted["c"];
        assert_eq!(c.1.len(), 1);
        assert!(c.1.contains(&OrderedFloat(5.0)));

        let subtracted = list2.difference(&list1);
        assert_eq!(subtracted.len(), 3);
        let a = &subtracted["a"];
        assert_eq!(a.1.len(), 1);
        assert!(a.1.contains(&OrderedFloat(3.0)));
        let b = &subtracted["b"];
        assert_eq!(b.1.len(), 1);
        assert!(b.1.contains(&OrderedFloat(5.0)));
        let c = &subtracted["c"];
        assert_eq!(c.1.len(), 1);
        assert!(c.1.contains(&OrderedFloat(7.0)));
    }

    #[test]
    fn test_difference_with_empy_set() {
        let mut list1 = NamedValuesCollection::new();
        list1.insert(
            "a".to_string(),
            (
                "cat".to_string(),
                vec![1.0, 2.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "b".to_string(),
            (
                "cat".to_string(),
                vec![3.0, 4.0].into_iter().map(OrderedFloat).collect(),
            ),
        );
        list1.insert(
            "c".to_string(),
            (
                "cat".to_string(),
                vec![5.0, 6.0].into_iter().map(OrderedFloat).collect(),
            ),
        );

        let list2 = NamedValuesCollection::new();

        let subtracted = list1.difference(&list2);
        assert_eq!(subtracted.len(), 3);
        let a = &subtracted["a"];
        assert_eq!(a.1.len(), 2);
        assert!(a.1.contains(&OrderedFloat(1.0)));
        assert!(a.1.contains(&OrderedFloat(2.0)));
        let b = &subtracted["b"];
        assert_eq!(b.1.len(), 2);
        assert!(b.1.contains(&OrderedFloat(3.0)));
        assert!(b.1.contains(&OrderedFloat(4.0)));
        let c = &subtracted["c"];
        assert_eq!(c.1.len(), 2);
        assert!(c.1.contains(&OrderedFloat(5.0)));
        assert!(c.1.contains(&OrderedFloat(6.0)));

        let subtracted = list2.difference(&list1);
        assert_eq!(subtracted.len(), 3);
        let a = &subtracted["a"];
        assert_eq!(a.1.len(), 0);
        let b = &subtracted["b"];
        assert_eq!(b.1.len(), 0);
        let c = &subtracted["c"];
        assert_eq!(c.1.len(), 0);
    }
}
