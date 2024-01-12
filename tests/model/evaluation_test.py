import unittest

from source.model.evaluation import (
    EvaluationMetrics,
)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        # Define some sample data for testing
        self.y_true = [1, 0, 1, 1, 0, 1, 0, 1]
        self.y_pred = [1, 0, 1, 0, 0, 1, 1, 1]

    def test_specificity(self):
        evaluator = EvaluationMetrics(self.y_true, self.y_pred, METRICS=["specificity"])
        specificity_result = evaluator.specificity
        self.assertIsInstance(specificity_result, float)

    def test_f1(self):
        evaluator = EvaluationMetrics(self.y_true, self.y_pred, METRICS=["f1"])
        f1_result = evaluator.f1
        self.assertIsInstance(f1_result, float)

    def test_recall(self):
        evaluator = EvaluationMetrics(self.y_true, self.y_pred, METRICS=["recall"])
        recall_result = evaluator.recall
        self.assertIsInstance(recall_result, float)

    def test_str_output(self):
        evaluator = EvaluationMetrics(
            self.y_true, self.y_pred, METRICS=["f1", "recall"]
        )
        str_output = str(evaluator)
        self.assertIsInstance(str_output, str)
