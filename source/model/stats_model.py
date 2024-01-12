import statsmodels.api as sm


class StatsModel:
    def __init__(self, **kwargs):
        self.architecture = kwargs.get("ARCHITECTURE", "sm.Logit")
        self.model = eval(self.architecture)
        self.fitted_model = None

    def fit(self, X, y, show_summary=False):
        model = self.model(y, X)
        trained_model = model.fit()
        if show_summary:
            print(trained_model.summary())
        self.fitted_model = trained_model

    def predict(self, X):
        y_pred = self.fitted_model.predict(X)
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred
