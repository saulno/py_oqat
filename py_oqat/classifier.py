import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from simple_oqat import oqat_with_aco

from .config_algorithms import Config
from .oqat_model import OQATModel
from .selector import Selector


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
        self.column_types = None
        self.discretizer = None

    def model_from_json(self, json):
        for learning_class, model in json.items():
            lc = int(learning_class)
            self.model[lc] = {"oqat_model": OQATModel(model["oqat_model"])}
            self.model[lc]["score"] = model["score"]
            self.model[lc]["cnf_weights"] = model["cnf_weights"]
            self.model[lc]["cnf_weights_norm"] = [w / sum(model["cnf_weights"]) for w in model["cnf_weights"]]
            self.classes.append(lc)


    def fit(self, X: list[list[float]], y: list[bool], column_names: list[str], column_types: list[str], n_discrete_bins: int = 1, learn_classes: list[int] = None):
        # Create a subset for validating the quality of the model
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X_train, self.y_train = X_train, y_train
        self.column_types = column_types

        # Discretize numerical columns
        continuous_columns_idx = [i for i in range(len(column_types)) if column_types[i] == "num"]
        if len(continuous_columns_idx) > 0:
            continuous_columns = X_train[:, continuous_columns_idx]
            est = KBinsDiscretizer(n_bins=n_discrete_bins, encode='ordinal', strategy='kmeans')
            est.fit(continuous_columns)
            X_train[:, continuous_columns_idx] = est.transform(continuous_columns)
            est_bin_edges = est.bin_edges_
            self.discretizer = est

        # for every learning class in y, create a model
        self.classes = learn_classes if not learn_classes is None else sorted(set(y))
        for learning_class in self.classes:
            model, weights = self.oqat_function[self.heuristic](X_train, y_train, learning_class, column_names, column_types, self.heuristic_config)
            model = OQATModel(model)

            # Transform the model for numerical columns into intervals
            if len(continuous_columns_idx) > 0:
                numeric_column_names = [column_name for i, column_name in enumerate(column_names) if column_types[i] == "num"]
                disjunctive_clauses_list = model.model.clauses
                for disjuntive_clause in disjunctive_clauses_list:
                    selectors_list = disjuntive_clause.clauses
                    for numeric_column in numeric_column_names:
                        bins_for_numeric_column = [selector.value[0] for selector in selectors_list if selector.column_name == numeric_column]
                        if len(bins_for_numeric_column) > 0:
                            contiguous_bins = OQATClassifier.collect_contiguous_bins(bins_for_numeric_column)
                            bin_edges = est_bin_edges[numeric_column_names.index(numeric_column)]
                            new_selectors = OQATClassifier.create_interval_selectors(numeric_column, contiguous_bins, bin_edges)
                            final_selectors = [selector for selector in disjuntive_clause.clauses if selector.column_name != numeric_column]
                            final_selectors.extend(new_selectors)
                            disjuntive_clause.clauses = final_selectors

            self.model[learning_class] = {"oqat_model": model}
            self.model[learning_class]["cnf_weights"] = weights
            self.model[learning_class]["cnf_weights_norm"] = [w / sum(weights) for w in weights]
            # print("Model for class", learning_class, "created")

            y_pred_val = model.fast_predict(X_val, column_names)
            y_real_val = [learning_class == y_val[i] for i in range(len(y_val))]
            score = sum([1 for i in range(len(y_pred_val)) if y_pred_val[i] == y_real_val[i]]) / len(y_pred_val)
            self.model[learning_class]["score"] = score

            # print("Score:", score)
    
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
        continuous_columns_idx = [i for i in range(len(self.column_types)) if self.column_types[i] == "num"]
        if len(continuous_columns_idx) == 0:
            return sum([self.dissimilarity(x1, x2) for x2 in X]) / len(X)
        
        continuous_columns = [x1[i] for i in continuous_columns_idx]
        discretized_columns = self.discretizer.transform([continuous_columns])[0]
        x1_copy = x1.copy() 
        for i in range(len(continuous_columns_idx)):
            x1_copy[continuous_columns_idx[i]] = discretized_columns[i]

        return sum([self.dissimilarity(x1_copy, x2) for x2 in X]) / len(X)
    
    def confusion_matrix(self, y_pred: list[float], y_test: list[float]) -> float:
        confusion_matrix = [[0 for _ in range(len(self.classes))] for _ in range(len(self.classes))]
        for actual in range(len(confusion_matrix)):
            for predicted in range(len(confusion_matrix[actual])):
                confusion_matrix[actual][predicted] = sum([1 for i in range(len(y_pred)) if y_pred[i] == predicted and y_test[i] == actual])
        
        return confusion_matrix
    
    def score(self, y_pred: list[float], y_test: list[float]) -> float:
        confusion_matrix = self.confusion_matrix(y_pred, y_test)
        return sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))]) / sum([sum(confusion_matrix[i]) for i in range(len(confusion_matrix))])
    
    def collect_contiguous_bins(bins: list[float]) -> list[list[float]]:
        """
        Collects contiguous bins into a list of lists.
        """
        bins = sorted(bins)
        contiguous_bins = []
        contiguous_segment = []
        for bin_ in bins:
            bin_ = int(bin_)
            if len(contiguous_segment) == 0:
                contiguous_segment.append(bin_)
            elif contiguous_segment[-1] + 1 == bin_:
                contiguous_segment.append(bin_)
            else:
                contiguous_bins.append(contiguous_segment)
                contiguous_segment = [bin_]
        
        if len(contiguous_segment) > 0:
            contiguous_bins.append(contiguous_segment)
        
        return contiguous_bins
    
    def create_interval_selectors(column_name: str, contiguous_bins: list[list[float]], bin_edges: list[float]) -> list["Selector"]:
        """
        Creates a list of selectors for a given column.
        """
        selectors = []
        left_bound, right_bound = 0, len(bin_edges) - 1
        for contiguous in contiguous_bins:
            left, right = contiguous[0], contiguous[-1]
            if left == left_bound and right == right_bound:
                selectors.append(Selector(column_name, -np.inf, np.inf))
            elif left == left_bound:
                selectors.append(Selector(column_name, -np.inf, bin_edges[right + 1]))
            elif right == right_bound:
                selectors.append(Selector(column_name, bin_edges[left], np.inf))
            else:
                selectors.append(Selector(column_name, bin_edges[left], bin_edges[right + 1]))
        
        return selectors





