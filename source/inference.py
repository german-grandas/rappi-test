from .datasets import TitanicDataset
from .model import StatsModel


def inference_from_data(data, configuration):
    data_config = configuration.get("DATA", None)
    if not data_config:
        raise Exception("Not data provided")

    dataset_type = data_config.get("TYPE", TitanicDataset)
    dataset = eval(dataset_type)(**data_config)

    model_config = configuration.get("MODEL", None)
    if not model_config:
        raise Exception("Not model configuration provided")

    model_name = model_config.get("NAME", StatsModel)
    model = eval(model_name)(**model_config)

    X = dataset.create_dataset_from_data(data)
    y_pred = model.predict(X)
    return y_pred
