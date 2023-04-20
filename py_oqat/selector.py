import numpy as np

from .clause import Clause


class Selector(Clause):
    def __init__(self, column_name: str, value: float, value_2: float = None, operator: str = None):
        self.column_name = column_name
        if value_2 is None:
            self.operator = operator
            self.operator_function = lambda x, y: x == y if operator == "=" else x < y if operator == "<" else x >= y
            self.value = [value]
            self.is_interval = False
        elif value == -np.inf:
            self.operator = "<"
            self.operator_function = lambda x, y: x < y 
            self.value = [value_2]
            self.is_interval = False
        elif value_2 == np.inf:
            self.operator = ">="
            self.operator_function = lambda x, y: x >= y
            self.value = [value]
            self.is_interval = False
        else:
            self.operator = [">=", "<"]
            self.operator_function = lambda x, v1, v2: x >= v1 and x < v2
            self.value = [value, value_2]
            self.is_interval = True

    def predict(self, X: list[list[float]], column_names: list[str]) -> list[bool]:
        column_index = column_names.index(self.column_name)
        return [self.operator_function(X[i][column_index], *self.value) for i in range(len(X))]

    def fast_predict(self, X: list[list[float]], column_names: list[str]) -> list[float]:
        column_index = column_names.index(self.column_name)
        arr = np.array([1 if self.operator_function(X[i][column_index], *self.value) else 0 for i in range(len(X))])
        return arr

    def __repr__(self):
        if self.is_interval:
            return f"[{self.value[0]:.2f}{self.operator[0]}{self.column_name}{self.operator[1]}{self.value[1]:.2f}]"
        
        return f"[{self.column_name}{self.operator}{self.value[0]:.2f}]"
