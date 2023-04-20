import numpy as np

from .clause import Clause
from .disjunctive_clause import DisjunctiveClause


class ConjunctiveClause(Clause):
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses

    def from_list_disjunctive_clauses(clauses: list[list[tuple[str, str, float]]]) -> 'ConjunctiveClause':
        return ConjunctiveClause([DisjunctiveClause.from_list_selectors(clause) for clause in clauses])

    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        return [all([clause.predict(X, column_names)[i] for clause in self.clauses]) for i in range(len(X))]
    
    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[float]:
        prod = np.ones(len(X))
        for clause in self.clauses:
            prod *= clause.fast_predict(X, column_names)
        return prod
    
    def precision_predict(self, X: list[list[float]], column_names: list[str], cnf_weights: list[int]) -> list[float]:
        sum = np.zeros(len(X))
        for clause_idx in range(len(self.clauses)):
            sum += cnf_weights[clause_idx] * self.clauses[clause_idx].fast_predict(X, column_names)
        return sum

    def __repr__(self):
        return f"({' ∧ '.join([str(clause) for clause in self.clauses])})"
