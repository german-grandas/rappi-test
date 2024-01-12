from sklearn.metrics import f1_score, recall_score, confusion_matrix


class EvaluationMetrics:
    def __init__(self, y_true, y_pred, **kwargs) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = kwargs.get("METRICS")

    @property
    def specificity(self):
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        specificity_value = tn / (tn + fp)
        return specificity_value

    @property
    def f1(self):
        return f1_score(self.y_true, self.y_pred)

    @property
    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def __str__(self):
        base_str = ""
        for metric in self.metrics:
            base_str += f" {metric}: {getattr(self, metric)}\n"
        return base_str
