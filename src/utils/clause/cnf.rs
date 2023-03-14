use std::fmt;

use crate::utils::data_handling::row::Row;

use super::disjunctive_clause::DisjunctiveClause;

pub struct Cnf {
    pub clauses: Vec<DisjunctiveClause>,
}

impl fmt::Display for Cnf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for clause in &self.clauses {
            write!(f, "{}", clause)?;
            if clause != self.clauses.last().unwrap() {
                write!(f, " ∧ ")?;
            }
        }
        Ok(())
    }
}

impl Default for Cnf {
    fn default() -> Self {
        Self::new()
    }
}

impl Cnf {
    pub fn new() -> Cnf {
        Cnf { clauses: vec![] }
    }

    pub fn accepts_row(&self, row: &Row) -> bool {
        for clause in &self.clauses {
            if !clause.accepts_row(row) {
                return false;
            }
        }
        true
    }

    pub fn to_export_format(&self) -> Vec<Vec<(String, String, f64)>> {
        let mut simple_format = Vec::new();
        for clause in &self.clauses {
            simple_format.push(clause.to_simple_format());
        }
        simple_format
    }
}
