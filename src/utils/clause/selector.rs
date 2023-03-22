use std::fmt;

use ordered_float::OrderedFloat;

use crate::utils::data_handling::row::Row;

#[derive(PartialEq, Clone)]
pub enum Selector {
    Eq(String, f64),
    Leq(String, f64),
    Geq(String, f64),
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Selector::Eq(attr, val) => write!(f, "[{}={}]", attr, val),
            Selector::Leq(attr, val) => write!(f, "[{}<={}]", attr, val),
            Selector::Geq(attr, val) => write!(f, "[{}>={}]", attr, val),
        }
    }
}

impl Selector {
    pub fn new_eq(attr: String, value: f64) -> Selector {
        Selector::Eq(attr, value)
    }

    pub fn new_leq(attr: String, value: f64) -> Selector {
        Selector::Leq(attr, value)
    }

    pub fn new_geq(attr: String, value: f64) -> Selector {
        Selector::Geq(attr, value)
    }

    fn get_column_name(&self) -> String {
        match self {
            Selector::Eq(attr, _) => attr.clone(),
            Selector::Leq(attr, _) => attr.clone(),
            Selector::Geq(attr, _) => attr.clone(),
        }
    }

    fn accepts_value(&self, value: f64) -> bool {
        match self {
            Selector::Eq(_, val) => OrderedFloat(value) == *val,
            Selector::Leq(_, val) => OrderedFloat(value) <= OrderedFloat(*val),
            Selector::Geq(_, val) => OrderedFloat(value) >= OrderedFloat(*val),
        }
    }

    pub fn accepts_row(&self, row: &Row) -> bool {
        for (column_name, (_, attr)) in &row.attributes {
            if column_name == &self.get_column_name() {
                for value in attr.iter() {
                    if self.accepts_value(**value) {
                        return true;
                    }
                }
            }
        }

        false
    }

    pub fn to_simple_format(&self) -> (String, String, f64) {
        match self {
            Selector::Eq(attr, val) => (attr.clone(), String::from("="), *val),
            Selector::Leq(attr, val) => (attr.clone(), String::from("<="), *val),
            Selector::Geq(attr, val) => (attr.clone(), String::from(">="), *val),
        }
    }
}
