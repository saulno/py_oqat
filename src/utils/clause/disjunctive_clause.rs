use std::fmt;

use crate::utils::data_handling::{named_values_collection::NamedValuesCollection, row::Row};

use super::selector::Selector;

#[derive(PartialEq)]
pub struct DisjunctiveClause {
    pub selectors: Vec<Selector>,
}

impl fmt::Display for DisjunctiveClause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "( ")?;
        for selector in &self.selectors {
            write!(f, "{}", selector)?;
            if selector != self.selectors.last().unwrap() {
                write!(f, " ∨ ")?;
            }
        }
        write!(f, " )")
    }
}

impl DisjunctiveClause {
    pub fn new(selectors: Vec<Selector>) -> DisjunctiveClause {
        DisjunctiveClause { selectors }
    }

    pub fn from_named_values_set_list(
        named_values_set_list: &NamedValuesCollection,
    ) -> DisjunctiveClause {
        let mut selectors = Vec::new();
        for (col_name, (_data_type, values_set)) in named_values_set_list {
            for value in values_set.iter() {
                selectors.push(Selector::Eq(col_name.clone(), **value));
            }
        }
        DisjunctiveClause::new(selectors)
    }

    pub fn accepts_row(&self, row: &Row) -> bool {
        for selector in &self.selectors {
            if selector.accepts_row(row) {
                return true;
            }
        }
        false
    }

    pub fn to_simple_format(&self) -> Vec<(String, String, f64)> {
        let mut simple_format = Vec::new();
        for selector in &self.selectors {
            simple_format.push(selector.to_simple_format());
        }
        simple_format
    }
}
