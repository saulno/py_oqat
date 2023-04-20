import numpy as np

from .clause import Clause
from .selector import Selector


class DisjunctiveClause(Clause):
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses

    def from_list_selectors(clauses: list[tuple[str, str, float]]) -> 'DisjunctiveClause':
        return DisjunctiveClause([Selector(clause[0], clause[2], operator=clause[1]) for clause in clauses])

    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        return [any([clause.predict(X, column_names)[i] for clause in self.clauses]) for i in range(len(X))]
        # return [any([clause.predict(X, column_names) for clause in self.clauses]) for i in range(len(X))]
    
    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[float]:
        sum = np.zeros(len(X))
        for clause in self.clauses:
            sum += clause.fast_predict(X, column_names)
        
        # maximum value is 1
        return np.minimum(sum, 1)

    def __repr__(self):
        return f"({' ∨ '.join([str(clause) for clause in self.clauses])})"
