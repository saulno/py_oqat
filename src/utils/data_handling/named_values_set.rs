use std::{collections::HashSet, fmt};

use ordered_float::OrderedFloat;

#[derive(Clone, Debug, PartialEq)]
pub struct NamedValuesSet {
    pub column_name: String,
    pub values: HashSet<OrderedFloat<f64>>,
}

impl Default for NamedValuesSet {
    fn default() -> Self {
        Self::new()
    }
}

impl NamedValuesSet {
    pub fn new() -> NamedValuesSet {
        NamedValuesSet {
            column_name: String::new(),
            values: HashSet::new(),
        }
    }

    pub fn to_export_format(&self) -> (String, Vec<f64>) {
        (
            self.column_name.clone(),
            self.values.iter().map(|x| x.into_inner()).collect(),
        )
    }

    pub fn from_ordered_float_vec(
        column_name: String,
        values: Vec<OrderedFloat<f64>>,
    ) -> NamedValuesSet {
        NamedValuesSet {
            column_name,
            values: values.into_iter().collect(),
        }
    }

    pub fn from_f64_vec(column_name: String, values: Vec<f64>) -> NamedValuesSet {
        NamedValuesSet {
            column_name,
            values: values.into_iter().map(OrderedFloat).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl fmt::Display for NamedValuesSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {{", self.column_name)?;
        for value in &self.values {
            write!(f, "{}", value)?;
            if value != self.values.iter().last().unwrap() {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}
