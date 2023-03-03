from abc import ABC, abstractmethod
import random

from config_algorithms import Config
from simple_oqat import oqat_with_aco


class OQATClassifier():
    def __init__(self, collision_strategy: str, heuristic: str, heuristic_config: Config):
        self.collision_strategy = collision_strategy
        self.heuristic = heuristic
        self.heuristic_config = heuristic_config
        self.model= {}
        self.oqat_function = {
            "aco": oqat_with_aco
        }
        self.classes = 0

    def fit(self, X: list[list[float]], y: list[bool], column_names: list[str], column_types: list[str]):
        # for every learning class in y, create a model
        self.classes = sorted(set(y))
        for learning_class in self.classes:
        # for learning_class in self.classes[:-1]:
            model = self.oqat_function[self.heuristic](X, y, learning_class, column_names, column_types, self.heuristic_config)
            model = OQATModel(model)
            self.model[learning_class] = model
            print("Model for class", learning_class, "created")
    
    def predict(self, X: list[list[float]], column_names: list[str]) -> list[set[float]]:
        predictions = [set() for _ in range(len(X))]
        for learning_class, model in self.model.items():
            y_pred = model.predict(X, column_names)
            for i in range(len(y_pred)):
                if y_pred[i]:
                    predictions[i].add(learning_class)
        
        default_class = -1
        # default_class = self.classes[-1]
        if self.collision_strategy == "random":
            predictions = [random.sample(sorted(predictions[i]), 1)[0] if len(predictions[i]) > 0 else default_class for i in range(len(predictions))]
        
        return predictions
    
    def confusion_matrix(self, y_pred: list[float], y_test: list[float]) -> float:
        confusion_matrix = [[0 for _ in range(len(self.classes))] for _ in range(len(self.classes))]
        for actual in range(len(confusion_matrix)):
            for predicted in range(len(confusion_matrix[actual])):
                confusion_matrix[actual][predicted] = sum([1 for i in range(len(y_pred)) if y_pred[i] == predicted and y_test[i] == actual])
        
        return confusion_matrix
    
    def score(self, y_pred: list[float], y_test: list[float]) -> float:
        confusion_matrix = self.confusion_matrix(y_pred, y_test)
        return sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))]) / sum([sum(confusion_matrix[i]) for i in range(len(confusion_matrix))])
                

class OQATModel():
    def __init__(self, model: list[list[tuple[str, str, float]]]):
        self.model = ConjunctiveClause.from_list_disjunctive_clauses(model)
    
    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        return self.model.predict(X, column_names)
    
    def __repr__(self):
        return str(self.model)


class Clause(ABC):
    @abstractmethod
    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        pass

    @abstractmethod
    def __repr__(self):
        pass

class Selector(Clause):
    def __init__(self, column_name: str, operator: str, value: float):
        self.column_name = column_name
        self.operator = operator
        self.operator_function = lambda x, y: x == y if operator == "=" else x <= y if operator == "<=" else x >= y
        self.value = value

    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        column_index = column_names.index(self.column_name)
        return [self.operator_function(X[i][column_index], self.value) for i in range(len(X))]

    def __repr__(self):
        return f"[{self.column_name}{self.operator}{self.value}]"

class DisjunctiveClause(Clause):
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses

    def from_list_selectors(clauses: list[tuple[str, str, float]]) -> 'DisjunctiveClause':
        return DisjunctiveClause([Selector(*clause) for clause in clauses])

    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        return [any([clause.predict(X, column_names)[i] for clause in self.clauses]) for i in range(len(X))]

    def __repr__(self):
        return f"({' ∨ '.join([str(clause) for clause in self.clauses])})"
    
class ConjunctiveClause(Clause):
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses

    def from_list_disjunctive_clauses(clauses: list[list[tuple[str, str, float]]]) -> 'ConjunctiveClause':
        return ConjunctiveClause([DisjunctiveClause.from_list_selectors(clause) for clause in clauses])

    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        return [all([clause.predict(X, column_names)[i] for clause in self.clauses]) for i in range(len(X))]

    def __repr__(self):
        return f"({' ∧ '.join([str(clause) for clause in self.clauses])})"