use std::fmt;

use crate::utils::data_handling::row::Row;

use super::disjunctive_clause::DisjunctiveClause;

pub struct Cnf {
    pub clauses: Vec<DisjunctiveClause>,
    pub weights: Vec<usize>,
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
        Cnf { clauses: vec![], weights: vec![] }
    }

    pub fn accepts_row(&self, row: &Row) -> bool {
        for clause in &self.clauses {
            if !clause.accepts_row(row) {
                return false;
            }
        }
        true
    }

    pub fn to_export_format(&self) -> (Vec<Vec<(String, String, f64)>>, Vec<usize>) {
        let mut simple_format = Vec::new();
        for clause in &self.clauses {
            simple_format.push(clause.to_simple_format());
        }
        (simple_format, self.weights.clone())
    }
    
    pub fn push_clause(&mut self, clause: DisjunctiveClause, weight: usize) {
        for (i, other_clause) in self.clauses.iter().enumerate() {
            if clause.equals(other_clause) {
                self.weights[i] += weight;
                return;
            }
        }
        self.clauses.push(clause);
        self.weights.push(weight);
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::clause::{selector::Selector, disjunctive_clause::DisjunctiveClause, cnf::Cnf};

    #[test]
    fn test_add_clause_no_collision() {
        let mut cnf = Cnf::new();
        let clause_1 = DisjunctiveClause::new(vec![Selector::Eq("a".to_string(), 1.0)]);
        let clause_2 = DisjunctiveClause::new(vec![Selector::Eq("b".to_string(), 1.0)]);
        let clause_3 = DisjunctiveClause::new(vec![Selector::Eq("c".to_string(), 1.0)]);

        cnf.push_clause(clause_1, 10);
        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.weights.len(), 1);
        assert_eq!(cnf.weights[0], 10);

        cnf.push_clause(clause_2, 20);
        assert_eq!(cnf.clauses.len(), 2);
        assert_eq!(cnf.weights.len(), 2);
        assert_eq!(cnf.weights[0], 10);
        assert_eq!(cnf.weights[1], 20);

        cnf.push_clause(clause_3, 30);
        assert_eq!(cnf.clauses.len(), 3);
        assert_eq!(cnf.weights.len(), 3);
        assert_eq!(cnf.weights[0], 10);
        assert_eq!(cnf.weights[1], 20);
        assert_eq!(cnf.weights[2], 30);
    }

    #[test]
    fn test_add_clause_with_collison() {
        let mut cnf = Cnf::new();
        let clause_1 = DisjunctiveClause::new(vec![Selector::Eq("a".to_string(), 1.0)]);
        let clause_2 = DisjunctiveClause::new(vec![Selector::Eq("a".to_string(), 1.0)]);
        let clause_3 = DisjunctiveClause::new(vec![Selector::Eq("b".to_string(), 1.0)]);
        let clause_4 = DisjunctiveClause::new(vec![Selector::Eq("c".to_string(), 1.0)]);
        let clause_5 = DisjunctiveClause::new(vec![Selector::Eq("c".to_string(), 1.0)]);

        cnf.push_clause(clause_1, 10);
        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.weights.len(), 1);
        assert_eq!(cnf.weights[0], 10);

        cnf.push_clause(clause_2, 20);
        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.weights.len(), 1);
        assert_eq!(cnf.weights[0], 30);

        cnf.push_clause(clause_3, 30);
        assert_eq!(cnf.clauses.len(), 2);
        assert_eq!(cnf.weights.len(), 2);
        assert_eq!(cnf.weights[0], 30);
        assert_eq!(cnf.weights[1], 30);

        cnf.push_clause(clause_4, 40);
        assert_eq!(cnf.clauses.len(), 3);
        assert_eq!(cnf.weights.len(), 3);
        assert_eq!(cnf.weights[0], 30);
        assert_eq!(cnf.weights[1], 30);
        assert_eq!(cnf.weights[2], 40);

        cnf.push_clause(clause_5, 50);
        assert_eq!(cnf.clauses.len(), 3);
        assert_eq!(cnf.weights.len(), 3);
        assert_eq!(cnf.weights[0], 30);
        assert_eq!(cnf.weights[1], 30);
        assert_eq!(cnf.weights[2], 90);
    }
}
