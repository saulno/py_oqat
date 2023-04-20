import numpy as np

from .conjunctive_clause import ConjunctiveClause


class OQATModel():
    def __init__(self, model: list[list[tuple[str, str, float]]]):
        self.model = ConjunctiveClause.from_list_disjunctive_clauses(model)
    
    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        return self.model.predict(X, column_names)
    
    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        numeric = self.model.fast_predict(X, column_names)
        return numeric == np.ones(len(numeric))
    
    def precision_predict(self, X: list[list[float]], column_names: list[str], cnf_weights: list[int]) -> list[float]:
        return self.model.precision_predict(X, column_names, cnf_weights)
    
    def __repr__(self):
        return str(self.model)