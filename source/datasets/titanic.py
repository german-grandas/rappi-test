import numpy as np
import pandas as pd


class TitanicDataset:
    def __init__(self, **kwargs) -> None:
        self.train = self.load_and_create_dataset("train", **kwargs)
        self.test = self.load_and_create_dataset("test", **kwargs)
        self.config = kwargs

    def load_and_create_dataset(self, type, **kwargs):
        data_path = kwargs.get("PATH")
        if not data_path:
            raise Exception("Not data path provided")

        if type == "train":
            df = pd.read_csv(f"{data_path}/train.csv")
        else:
            df = pd.read_csv(f"{data_path}/test.csv")
            gender_submission_df = pd.read_csv(f"{data_path}/gender_submission.csv")
            df = pd.merge(df, gender_submission_df, on="PassengerId", how="right")

        df = df.dropna()
        df["Age"] = df["Age"].apply(lambda x: np.ceil(x))
        df["Sex"] = df["Sex"].astype("category")
        df["Embarked"] = df["Embarked"].astype("category")

        df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True)
        return df

    def get_model_ready_dataset(self, dataset_type):
        objective_variable = self.config.get("OBJECTIVE_VARIABLE")
        if not objective_variable:
            raise Exception("Objective variable not defined")

        if dataset_type == "train":
            X = self.train.drop(objective_variable, axis=1)
            y = self.train[[objective_variable]]
        else:
            X = self.test.drop(objective_variable, axis=1)
            y = self.test[[objective_variable]]
        X = pd.get_dummies(X).astype(float)
        return X, y

    @staticmethod
    def create_dataset_from_data(data):
        return pd.DataFrame(data)
