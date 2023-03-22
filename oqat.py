from abc import ABC, abstractmethod
import random
import numpy as np

from sklearn.model_selection import train_test_split

from config_algorithms import Config
from simple_oqat import oqat_with_aco


class OQATClassifier():
    def __init__(self, collision_strategy: str, null_strategy: str, heuristic: str, heuristic_config: Config):
        self.collision_strategy = collision_strategy
        self.null_strategy = null_strategy

        self.heuristic = heuristic
        self.heuristic_config = heuristic_config

        self.model= {}
        self.oqat_function = {
            "aco": oqat_with_aco
        }

        self.classes = []
        self.X_train, y_train = None, None

    def model_from_json(self, json):
        for learning_class, model in json.items():
            lc = int(learning_class)
            self.model[lc] = {"oqat_model": OQATModel(model["oqat_model"])}
            self.model[lc]["score"] = model["score"]
            self.model[lc]["cnf_weights"] = model["cnf_weights"]
            self.model[lc]["cnf_weights_norm"] = [w / sum(model["cnf_weights"]) for w in model["cnf_weights"]]
            self.classes.append(lc)


    def fit(self, X: list[list[float]], y: list[bool], column_names: list[str], column_types: list[str]):
        # Create a subset for validating the quality of the model
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X_train, self.y_train = X_train, y_train

        # for every learning class in y, create a model
        self.classes = sorted(set(y))
        for learning_class in self.classes:
            model, weights = self.oqat_function[self.heuristic](X_train, y_train, learning_class, column_names, column_types, self.heuristic_config)
            model = OQATModel(model)

            self.model[learning_class] = {"oqat_model": model}
            self.model[learning_class]["cnf_weights"] = weights
            self.model[learning_class]["cnf_weights_norm"] = [w / sum(weights) for w in weights]
            print("Model for class", learning_class, "created")

            y_pred_val = model.fast_predict(X_val, column_names)
            y_real_val = [learning_class == y_val[i] for i in range(len(y_val))]
            score = sum([1 for i in range(len(y_pred_val)) if y_pred_val[i] == y_real_val[i]]) / len(y_pred_val)
            self.model[learning_class]["score"] = score

            print("Score:", score)
    
    def predict(self, X: list[list[float]], column_names: list[str]) -> list[set[float]]:
        predictions = [set() for _ in range(len(X))]
        for learning_class, model in self.model.items():
            y_pred = model["oqat_model"].fast_predict(X, column_names)
            for i in range(len(y_pred)):
                if y_pred[i]:
                    predictions[i].add(learning_class)
        
        final_predictions = [None for _ in range(len(predictions))]

        default_class = -1
        for i in range(len(predictions)):
            if len(predictions[i]) == 0:
                if self.null_strategy == "weighted":
                    weighted_classes = {l_class: self.model[l_class]["oqat_model"].precision_predict(X[i:i+1], column_names, self.model[l_class]["cnf_weights"]) for l_class in self.classes}
                    final_predictions[i] = max(self.classes, key=lambda l_class: weighted_classes[l_class])
                elif self.null_strategy == "weighted_normalized":
                    weighted_classes = {l_class: self.model[l_class]["oqat_model"].precision_predict(X[i:i+1], column_names, self.model[l_class]["cnf_weights_norm"]) for l_class in self.classes}
                    final_predictions[i] = max(self.classes, key=lambda l_class: weighted_classes[l_class])
                elif self.null_strategy == "dissimilarity":
                    final_predictions[i] = min(self.classes, key=lambda l_class: self.dissimilarity_vec(X[i], self.X_train[self.y_train == l_class]))
                else:
                    final_predictions[i] = default_class
            elif len(predictions[i]) > 1:
                if self.collision_strategy == "best_score":
                    final_predictions[i] = max(predictions[i], key=lambda l_class: self.model[l_class]["score"])
                elif self.collision_strategy == "dissimilarity":
                    final_predictions[i] = min(predictions[i], key=lambda l_class: self.dissimilarity_vec(X[i], self.X_train[self.y_train == l_class]))
                else:
                    final_predictions[i] = list(predictions[i])
            else:
                final_predictions[i] = list(predictions[i])[0]

        # if self.collision_strategy == "weights":
        #     final_predictions = [{l_class: self.model[l_class]["oqat_model"].precision_predict(X[i:i+1], column_names, self.model[l_class]["cnf_weights"]) for l_class in self.classes} for i in range(len(predictions))] 
        # elif self.collision_strategy == "weights_norm":
        #     final_predictions = [{l_class: self.model[l_class]["oqat_model"].precision_predict(X[i:i+1], column_names, self.model[l_class]["cnf_weights_norm"]) for l_class in self.classes} for i in range(len(predictions))] 
        
        
        return final_predictions

    def dissimilarity(self, x1: list[float], x2:list[float]) -> float:
        """
        Calculates the disimilarity between two vectors.
        0 means that the vectors are equal. 1 means that the vectors are completely different.
        """
        return sum([1 if x1[i] != x2[i] else 0 for i in range(len(x1))]) / len(x1)
    
    def dissimilarity_vec(self, x1: list[float], X: list[list[float]]) -> float:
        """
        Calculates the disimilarity between a vector and a matrix (set of vectors).
        """
        return sum([self.dissimilarity(x1, x2) for x2 in X]) / len(X)
    
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
    
    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        numeric = self.model.fast_predict(X, column_names)
        return numeric == np.ones(len(numeric))
    
    def precision_predict(self, X: list[list[float]], column_names: list[str], cnf_weights: list[int]) -> list[float]:
        return self.model.precision_predict(X, column_names, cnf_weights)
    
    def __repr__(self):
        return str(self.model)


class Clause(ABC):
    @abstractmethod
    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        pass

    @abstractmethod
    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
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

    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[float]:
        column_index = column_names.index(self.column_name)
        arr = np.array([1 if self.operator_function(X[i][column_index], self.value) else 0 for i in range(len(X))])
        return arr

    def __repr__(self):
        return f"[{self.column_name}{self.operator}{self.value}]"

class DisjunctiveClause(Clause):
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses

    def from_list_selectors(clauses: list[tuple[str, str, float]]) -> 'DisjunctiveClause':
        return DisjunctiveClause([Selector(*clause) for clause in clauses])

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