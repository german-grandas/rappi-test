import unittest
from unittest.mock import patch, Mock

from source.model import StatsModel, EvaluationMetrics
from source.datasets import TitanicDataset
from source.train import (
    train_model,
)


class TestTrainModel(unittest.TestCase):
    @patch("source.model.StatsModel.fit")
    @patch("source.model.StatsModel.predict")
    @patch("source.datasets.TitanicDataset.get_model_ready_dataset")
    @patch("source.datasets.TitanicDataset.load_and_create_dataset")
    def test_train_model(
        self,
        mock_load_and_create_dataset,
        mock_get_model_ready_dataset,
        mock_predict,
        mock_fit,
    ):
        # Mocking dataset
        mock_dataset = Mock(spec=TitanicDataset)
        mock_load_and_create_dataset.return_value = mock_dataset
        mock_get_model_ready_dataset.return_value = (Mock(), Mock())

        # Mocking model
        mock_model = Mock(spec=StatsModel)
        mock_model_instance = mock_model.return_value
        mock_model_instance.fitted_model = Mock()
        mock_model.return_value = mock_model_instance

        # Mocking EvaluationMetrics
        mock_evaluation_metrics = Mock(spec=EvaluationMetrics)
        mock_evaluation_metrics_instance = mock_evaluation_metrics.return_value
        mock_evaluation_metrics.return_value = mock_evaluation_metrics_instance

        with patch("builtins.print"):
            configuration = {
                "DATA": {"TYPE": "TitanicDataset", "PATH": "./data/"},
                "MODEL": {"NAME": "StatsModel"},
                "EVALUATION": {"METRICS": ["f1", "recall"]},
            }
            train_model(configuration)
