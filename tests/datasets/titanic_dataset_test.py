import unittest

import pandas as pd
from source.datasets.titanic import (
    TitanicDataset,
)


class TestTitanicDataset(unittest.TestCase):
    def setUp(self):
        # Define some sample data paths for testing
        self.sample_data_path = "./data"
        self.sample_config = {
            "PATH": self.sample_data_path,
            "OBJECTIVE_VARIABLE": "Survived",
        }

    def test_load_and_create_dataset_train(self):
        dataset = TitanicDataset(**self.sample_config)
        train_df = dataset.train
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertTrue("Survived" in train_df.columns)

    def test_load_and_create_dataset_test(self):
        dataset = TitanicDataset(**self.sample_config)
        test_df = dataset.test
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertTrue("Survived" in test_df.columns)

    def test_get_model_ready_dataset_train(self):
        dataset = TitanicDataset(**self.sample_config)
        X_train, y_train = dataset.get_model_ready_dataset("train")
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.DataFrame)

    def test_get_model_ready_dataset_test(self):
        dataset = TitanicDataset(**self.sample_config)
        X_test, y_test = dataset.get_model_ready_dataset("test")
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.DataFrame)

    def test_create_dataset_from_data(self):
        data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
        result_df = TitanicDataset.create_dataset_from_data(data)
        self.assertIsInstance(result_df, pd.DataFrame)
