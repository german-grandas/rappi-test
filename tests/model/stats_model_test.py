import unittest

import numpy as np
import pandas as pd

from source.model.stats_model import StatsModel


class TestStatsModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X = pd.DataFrame(
            {"Feature1": np.random.rand(100), "Feature2": np.random.rand(100)}
        )
        y = np.random.randint(0, 2, size=100)

        self.model = StatsModel(ARCHITECTURE="sm.Logit")
        self.X_train = X
        self.y_train = y

    def test_fit_predict(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        self.assertIsNotNone(self.model.fitted_model)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_fit_predict_summary(self):
        self.model.fit(self.X_train, self.y_train, show_summary=True)
        self.assertIsNotNone(self.model.fitted_model)
