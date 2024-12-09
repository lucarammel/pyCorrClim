from typing import Callable, Optional

import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid


class Smoother:
    def __init__(self, time_column="time", value_column=None):
        self.time_column = time_column
        self.value_column = value_column
        self.status = 0

    def fit(self, timeseries: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        raise NotImplementedError("fit method must be implemented")

    def smooth(self, timeseries: pd.DataFrame):
        if self.status < 1:
            raise ValueError("Please fit the smoother before applying it.")
        return self.smooth_fun(timeseries)

    def fit_smooth(self, timeseries: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        self.fit(timeseries, y)
        return self.smooth(timeseries)

    def export(self, path: str):
        if not path.endswith(".pkl"):
            raise ValueError("File path should have extension .pkl")
        import joblib

        joblib.dump(self, path)


class ExponentialSmoother(Smoother):
    def __init__(self, alpha=0.2, N=20, granularity="step", **kwargs):
        super().__init__(**kwargs)
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if granularity not in ["step", "days"]:
            raise ValueError("granularity should be either 'days' or 'step'")
        self.alpha = alpha
        self.N = N
        self.granularity = granularity

    def fit(self, timeseries: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        self.value_column = timeseries.columns[1]  # Assuming the second column is the value column
        self.status = 1

    def smooth_fun(self, timeseries: pd.DataFrame):
        if self.granularity == "step":
            timeseries[self.value_column] = (
                timeseries[self.value_column].ewm(span=1 / self.alpha, adjust=False).mean()
            )
        elif self.granularity == "days":
            # Assuming timeseries has a 'time' column in datetime format
            timeseries["day"] = timeseries[self.time_column].dt.date
            timeseries["shifted"] = timeseries.groupby("day")[self.value_column].shift(1)
            timeseries["shifted"].fillna(timeseries[self.value_column], inplace=True)
            timeseries["smoothed"] = (
                timeseries[self.value_column].ewm(span=self.N, adjust=False).mean()
            )
        return timeseries


class DummySmoother(Smoother):
    def fit(self, timeseries: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        pass

    def smooth_fun(self, timeseries: pd.DataFrame):
        return timeseries


class GridSearchSmoother(Smoother):
    def __init__(
        self, grid: dict, smoother_class: Smoother, score: Callable = mean_squared_error, **kwargs
    ):
        super().__init__(**kwargs)
        self.grid = grid
        self.smoother_class = smoother_class
        self.score = score
        self.best_params = None
        self.best_smoother = None

    def fit(self, timeseries: pd.DataFrame, y: pd.DataFrame):
        best_score = float("inf")
        param_combinations = list(ParameterGrid(self.grid))
        for params in param_combinations:
            smoother = self.smoother_class(**params)
            smoother.fit(timeseries, y)
            smoothed = smoother.smooth(timeseries)
            score = self.score(smoothed[self.value_column], y[self.value_column])
            if score < best_score:
                best_score = score
                self.best_params = params
                self.best_smoother = smoother

    def smooth_fun(self, timeseries: pd.DataFrame):
        if not self.best_smoother:
            raise ValueError("Smoother is not fitted yet.")
        return self.best_smoother.smooth(timeseries)


class BayesianSmoother(Smoother):
    def __init__(
        self,
        bounds: dict,
        smoother_class: Smoother,
        score: Callable = mean_squared_error,
        n_iter=20,
        init_points=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.smoother_class = smoother_class
        self.score = score
        self.n_iter = n_iter
        self.init_points = init_points
        self.best_params = None
        self.best_smoother = None

    def fit(self, timeseries: pd.DataFrame, y: pd.DataFrame):
        def objective(**params):
            smoother = self.smoother_class(**params)
            smoother.fit(timeseries, y)
            smoothed = smoother.smooth(timeseries)
            return self.score(smoothed[self.value_column], y[self.value_column])

        result = minimize(objective, x0=[0.5] * len(self.bounds), bounds=list(self.bounds.values()))
        self.best_params = dict(zip(self.bounds.keys(), result.x))
        self.best_smoother = self.smoother_class(**self.best_params)
        self.best_smoother.fit(timeseries, y)

    def smooth_fun(self, timeseries: pd.DataFrame):
        if not self.best_smoother:
            raise ValueError("Smoother is not fitted yet.")
        return self.best_smoother.smooth(timeseries)


class MultiSmoother(Smoother):
    def __init__(self, smoothers, variables):
        """
        Initialize the MultiSmoother class.

        :param smoothers: List of Smoother objects
        :param variables: List of variables to smooth
        """
        if len(smoothers) >= 1:
            if not isinstance(smoothers, list):
                raise ValueError("Please provide a list of smoothers")

        for s in smoothers:
            if not isinstance(s, Smoother):
                raise ValueError("Input is not of type Smoother")

        if len(variables) != len(smoothers):
            raise ValueError(
                f"MultiSmoother has length variables: {len(variables)} and length smoothers: {len(smoothers)}. "
                "Provide a Smoother for each variable, in the correct order."
            )

        self.smoothers = smoothers
        self.variables = variables

    def fit_fun(self, timeseries, y):
        """
        Fit the smoothers to the timeseries.

        :param timeseries: The timeseries data (pandas DataFrame)
        :param y: The response timeseries to compare with
        """
        for i, smoother in enumerate(self.smoothers):
            print(f"Fitting using {type(smoother).__name__} ..")

            X = timeseries[["time", self.variables[i]]]
            smoother.fit(X, y)

    def smooth_fun(self, timeseries):
        """
        Smooth the timeseries using the smoothers.

        :param timeseries: The timeseries data (pandas DataFrame)
        :return: The smoothed timeseries (pandas DataFrame)
        """
        smoothed_timeseries = timeseries.copy()

        for i, smoother in enumerate(self.smoothers):
            print(f"Smoothing using {type(smoother).__name__} on {self.variables[i]}..")

            X = timeseries[["time", self.variables[i]]]
            smoothed_data = smoother.smooth(X)

            smoothed_timeseries[self.variables[i]] = smoothed_data

        return smoothed_timeseries

    def get_smoothers(self):
        """
        Get the list of smoothers.

        :return: List of smoothers
        """
        return self.smoothers
