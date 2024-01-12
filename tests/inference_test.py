import unittest
from unittest.mock import patch, Mock
from source.datasets import TitanicDataset
from source.model import StatsModel
from source.inference import inference_from_data


class TestInferenceFromData(unittest.TestCase):
    @patch("source.datasets.TitanicDataset.create_dataset_from_data")
    @patch("source.model.StatsModel.predict")
    def test_inference_from_data(self, mock_predict, mock_create_dataset_from_data):
        # Mocking dataset
        mock_dataset = Mock(spec=TitanicDataset)
        mock_create_dataset_from_data.return_value = mock_dataset

        # Mocking model
        mock_model = Mock(spec=StatsModel)
        mock_model_instance = mock_model.return_value
        mock_model.return_value = mock_model_instance

        # Mocking data and configuration
        data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
        configuration = {
            "DATA": {"TYPE": "TitanicDataset", "PATH": "./data"},
            "MODEL": {"NAME": "StatsModel"},
        }

        # Call the function to be tested
        result = inference_from_data(data, configuration)

        # Assertions
        self.assertEqual(result, mock_predict.return_value)
