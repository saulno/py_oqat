use std::collections::HashSet;

use ordered_float::OrderedFloat;

use super::named_values_set::NamedValuesSet;

pub type NamedValuesSetList = Vec<NamedValuesSet>;

pub trait SetOperationsTrait {
    fn new() -> NamedValuesSetList;
    fn is_empty(&self) -> bool;
    fn union(&self, other: &NamedValuesSetList) -> NamedValuesSetList;
    fn intersection(&self, other: &NamedValuesSetList) -> NamedValuesSetList;
    fn difference(&self, other: &NamedValuesSetList) -> NamedValuesSetList;
}

impl SetOperationsTrait for NamedValuesSetList {
    fn new() -> NamedValuesSetList {
        Vec::new()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn union(&self, other: &NamedValuesSetList) -> NamedValuesSetList {
        let mut result: Vec<NamedValuesSet> = self.clone();
        for (idx, current) in self.iter().enumerate() {
            let new = &other[idx];
            let unified_values_set: HashSet<OrderedFloat<f64>> =
                current.values.union(&new.values).cloned().collect();
            result[idx] = NamedValuesSet {
                column_name: new.column_name.clone(),
                values: unified_values_set,
            };
        }

        result
    }

    fn intersection(&self, other: &NamedValuesSetList) -> NamedValuesSetList {
        let mut result: Vec<NamedValuesSet> = self.clone();
        for (idx, current) in self.iter().enumerate() {
            let new = &other[idx];
            let intersected_values_set: HashSet<OrderedFloat<f64>> =
                current.values.intersection(&new.values).cloned().collect();
            result[idx] = NamedValuesSet {
                column_name: new.column_name.clone(),
                values: intersected_values_set,
            };
        }

        result
    }

    fn difference(&self, other: &NamedValuesSetList) -> NamedValuesSetList {
        let mut result: Vec<NamedValuesSet> = self.clone();
        for (idx, current) in self.iter().enumerate() {
            let new = &other[idx];
            let subtracted_values_set: HashSet<OrderedFloat<f64>> =
                current.values.difference(&new.values).cloned().collect();
            result[idx] = NamedValuesSet {
                column_name: new.column_name.clone(),
                values: subtracted_values_set,
            };
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union() {
        let mut list1 = NamedValuesSetList::new();
        list1.push(NamedValuesSet::from_f64_vec(
            "a".to_string(),
            vec![1.0, 2.0],
        ));
        list1.push(NamedValuesSet::from_f64_vec(
            "b".to_string(),
            vec![3.0, 4.0],
        ));
        list1.push(NamedValuesSet::from_f64_vec(
            "c".to_string(),
            vec![5.0, 6.0],
        ));

        let mut list2 = NamedValuesSetList::new();
        list2.push(NamedValuesSet::from_f64_vec(
            "a".to_string(),
            vec![2.0, 3.0],
        ));
        list2.push(NamedValuesSet::from_f64_vec(
            "b".to_string(),
            vec![4.0, 5.0],
        ));
        list2.push(NamedValuesSet::from_f64_vec(
            "c".to_string(),
            vec![6.0, 7.0],
        ));

        let unified = list1.union(&list2);
        assert_eq!(unified.len(), 3);
        assert_eq!(
            unified[0].values,
            vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]
                .into_iter()
                .collect()
        );
        assert_eq!(
            unified[1].values,
            vec![OrderedFloat(3.0), OrderedFloat(4.0), OrderedFloat(5.0)]
                .into_iter()
                .collect()
        );
        assert_eq!(
            unified[2].values,
            vec![OrderedFloat(5.0), OrderedFloat(6.0), OrderedFloat(7.0)]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn test_intersection() {
        let mut list1 = NamedValuesSetList::new();
        list1.push(NamedValuesSet::from_f64_vec(
            "a".to_string(),
            vec![1.0, 2.0],
        ));
        list1.push(NamedValuesSet::from_f64_vec(
            "b".to_string(),
            vec![3.0, 4.0],
        ));
        list1.push(NamedValuesSet::from_f64_vec(
            "c".to_string(),
            vec![5.0, 6.0],
        ));

        let mut list2 = NamedValuesSetList::new();
        list2.push(NamedValuesSet::from_f64_vec(
            "a".to_string(),
            vec![2.0, 3.0],
        ));
        list2.push(NamedValuesSet::from_f64_vec(
            "b".to_string(),
            vec![4.0, 5.0],
        ));
        list2.push(NamedValuesSet::from_f64_vec(
            "c".to_string(),
            vec![6.0, 7.0],
        ));

        let intersected = list1.intersection(&list2);
        assert_eq!(intersected.len(), 3);
        assert_eq!(
            intersected[0].values,
            vec![OrderedFloat(2.0)].into_iter().collect()
        );
        assert_eq!(
            intersected[1].values,
            vec![OrderedFloat(4.0)].into_iter().collect()
        );
        assert_eq!(
            intersected[2].values,
            vec![OrderedFloat(6.0)].into_iter().collect()
        );
    }

    #[test]
    fn test_difference() {
        let mut list1 = NamedValuesSetList::new();
        list1.push(NamedValuesSet::from_f64_vec(
            "a".to_string(),
            vec![1.0, 2.0],
        ));
        list1.push(NamedValuesSet::from_f64_vec(
            "b".to_string(),
            vec![3.0, 4.0],
        ));
        list1.push(NamedValuesSet::from_f64_vec(
            "c".to_string(),
            vec![5.0, 6.0],
        ));

        let mut list2 = NamedValuesSetList::new();
        list2.push(NamedValuesSet::from_f64_vec(
            "a".to_string(),
            vec![2.0, 3.0],
        ));
        list2.push(NamedValuesSet::from_f64_vec(
            "b".to_string(),
            vec![4.0, 5.0],
        ));
        list2.push(NamedValuesSet::from_f64_vec(
            "c".to_string(),
            vec![6.0, 7.0],
        ));

        let subtracted = list1.difference(&list2);
        assert_eq!(subtracted.len(), 3);
        assert_eq!(
            subtracted[0].values,
            vec![OrderedFloat(1.0)].into_iter().collect()
        );
        assert_eq!(
            subtracted[1].values,
            vec![OrderedFloat(3.0)].into_iter().collect()
        );
        assert_eq!(
            subtracted[2].values,
            vec![OrderedFloat(5.0)].into_iter().collect()
        );
    }
}
