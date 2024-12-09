import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.regression.glm import GLM
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools import add_constant

from corrclim.timeseries_model import TimeseriesModel


class GradDelta(TimeseriesModel):
    def __init__(
        self,
        formula="y ~ temperature",
        lm: str = "robust",
        n_shift: int = 168,
        granularity: str = "day",
        N_min: int = 30,
        weights=None,
    ):
        self.formula = formula
        self.N_min = N_min
        self.weights = weights
        self.lm = lm
        self.n_shift = n_shift
        self.granularity = granularity
        self.gradients = None
        self.model = None
        self._initialize_linear_model()

    def _initialize_linear_model(self):
        if self.lm == "robust":
            self.lm_func = RLM
        elif self.lm == "least squares":
            self.lm_func = GLM
        elif self.lm == "ridge":
            self.lm_func = Ridge
        else:
            raise ValueError(
                "Linear model not supported. Choose 'robust', 'least squares', or 'ridge'."
            )

    def fit_fun(self, X):
        X = self._get_timeseries(X)

        if self.granularity == "instant":
            X = X.groupby("instant").apply(self._fit_and_extract_coefs)
        else:
            X = self._fit_and_extract_coefs(X)

        self.gradients = X
        return self.gradients

    def _get_timeseries(self, X):
        # Assuming X is a DataFrame, ensure it's in the right format
        if isinstance(X, pd.DataFrame):
            return X
        else:
            raise TypeError("X must be a pandas DataFrame.")

    def _fit_and_extract_coefs(self, data):
        model = self._linear_model(data)
        coefs = model.params[1:].values  # excluding the intercept
        return pd.Series(coefs, index=self._get_explanatory_variables())

    def _linear_model(self, dt):
        if len(dt) < self.N_min:
            raise ValueError("Not enough observations for fitting")

        X = add_constant(dt[self._get_explanatory_variables()])
        y = dt["y"]

        if self.weights is not None:
            model = self.lm_func(y, X, weights=self.weights).fit()
        else:
            model = self.lm_func(y, X).fit()

        return model

    def _get_explanatory_variables(self):
        return self.formula.split("~")[1].strip().split()

    def predict_fun(self, model, X):
        X = self._get_timeseries(X)
        vars_ = self._get_explanatory_variables()
        X = X.merge(model, on="instant", suffixes=("", "_grad"))

        X["result"] = 0
        for var in vars_:
            var_grad = f"{var}_grad"
            X["result"] += X[var] * X[var_grad]

        return X["result"]

    def get_gradients(self):
        return self.gradients.copy()
