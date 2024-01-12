from .model import StatsModel, EvaluationMetrics
from .datasets import TitanicDataset


def train_model(configuration):
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

    X, y = dataset.get_model_ready_dataset("train")
    show_summary = model_config.get("SUMMARY", False)
    model.fit(X, y, show_summary)

    evaluation_config = configuration.get("EVALUATION", None)
    if evaluation_config:
        print("\nRunning evaluation")
        X_test, y_true = dataset.get_model_ready_dataset("test")
        y_pred = model.predict(X_test)

        evaluation = EvaluationMetrics(y_true, y_pred, **evaluation_config)
        print(evaluation)
