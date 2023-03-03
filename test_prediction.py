import unittest

from oqat import ConjunctiveClause, DisjunctiveClause, OQATClassifier, Selector

class TestPrediction(unittest.TestCase):

    def test_selector_predcit(self):
        selector = Selector("column1", "=", 1)
        X = [[1, 2], [2, 3], [1, 3]]
        column_names = ["column1", "column2"]
        assert selector.predict(X, column_names) == [True, False, True]

    def test_disjunctive_predict(self):
        disjunctive_clause = DisjunctiveClause([Selector("column1", "=", 1), Selector("column2", "=", 2)])
        X = [[1, 2], [2, 3], [1, 3], [2, 2], [2, 1], [3, 2]]
        column_names = ["column1", "column2"]
        assert disjunctive_clause.predict(X, column_names) == [True, False, True, True, False, True]

    def test_conjunctive_predict(self):
        conjunctive_clause = ConjunctiveClause([DisjunctiveClause([Selector("column1", "=", 1), Selector("column2", "=", 2)]), DisjunctiveClause([Selector("column1", "=", 2), Selector("column2", "=", 3)])])
        X = [[1, 2], [2, 3], [1, 3], [2, 2], [2, 1], [3, 2]]
        column_names = ["column1", "column2"]
        assert conjunctive_clause.predict(X, column_names) == [False, False, True, True, False, False]

    def test_general_1(self):
        X = [
                [1., 3., 3., 1.],
                [0., 0., 2., 3.],
                [2., 0., 3., 3.],
                [0., 0., 1., 3.],
                [1., 1., 2., 3.],
                [0., 3., 1., 2.],
                [0., 0., 3., 2.],
                [2., 3., 0., 2.],
                [2., 0., 3., 1.],
                [1., 1., 3., 2.],
                [0., 2., 1., 3.],
                [0., 3., 0., 0.],
                [2., 2., 3., 1.],
                [2., 3., 2., 1.],
                [2., 3., 0., 1.],
                [1., 3., 2., 0.],
                [2., 1., 1., 3.],

                [2., 2., 2., 2.],
            ]
        y = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 1]
        column_names = ["hobby", "age", "education", "marital"]
        model = ConjunctiveClause([
            DisjunctiveClause([
                Selector("age", "=", 3),
                Selector("education", "=", 3),
                Selector("marital", "=", 3)
            ])
        ])
        classifier = OQATClassifier(collision_strategy="random", heuristic="aco", heuristic_config=None)
        classifier.model = {2 : model}
        classifier.classes = [0, 1, 2]
        y_pred = classifier.predict(X, column_names)
        cf = classifier.confusion_matrix(y_pred, y)
        print(y_pred)
        print(cf)
        # assert all(predictions)