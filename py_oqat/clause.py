from abc import ABC, abstractmethod


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
