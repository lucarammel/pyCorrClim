import pandas as pd
from patsy import dmatrices
from pygam import LinearGAM, s

from corrclim.timeseries_dt import TimeseriesDT
from corrclim.timeseries_model.timeseries_model import TimeseriesModel


class GAM(TimeseriesModel):
    def __init__(
        self,
        formula="y ~ s(temperature) + s(posan) + jour_semaine + jour_ferie + ponts",
        by_instant=True,
        granularity="day",
        *args,
        **kwargs,
    ):
        self.formula = formula
        self.by_instant = by_instant
        self.granularity = granularity
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if self.by_instant:
            self.model = LinearGAM(
                s(0) + s(1)
            )  # Example: Implementing basis splines for continuous variables
        else:
            self.model = None

    def fit_fun(self, X: TimeseriesDT):
        """
        Fit the model to the timeseries data.

        :param X: DataFrame with the timeseries data
        :return: Fitted model
        """
        if self.by_instant:
            # Fit by "instant"
            X_grouped = X.groupby("instant").apply(self._fit_and_extract_coefs)
            return X_grouped
        else:
            y, X = dmatrices(self.formula, data=X, return_type="dataframe")
            self.model.fit(X, y)
            return self.model

    def _fit_and_extract_coefs(self, group):
        """
        Fit a model for a particular group and extract coefficients
        """
        y, X = dmatrices(self.formula, data=group, return_type="dataframe")
        self.model.fit(X, y)
        coefs = self.model.coef_
        return pd.Series(coefs, index=self._get_explanatory_variables())

    def _get_explanatory_variables(self):
        """Extracts explanatory variables from the formula"""
        return self.formula.split("~")[1].strip().split(" + ")

    def predict_fun(self, X: TimeseriesDT):
        """
        Predict using the fitted model.

        :param X: DataFrame with timeseries data
        :return: Predictions (as a list or array)
        """
        if self.by_instant:
            return self.model.predict(X)
        else:
            return self.model.predict(X).tolist()


class GamStd(GAM):
    def __init__(
        self,
        formula="y ~ s(temperature) + s(posan) + jour_semaine + jour_ferie + ponts",
        by_instant=False,
        granularity="day",
        *args,
        **kwargs,
    ):
        super().__init__(formula, by_instant, granularity, *args, **kwargs)

    def fit_fun(self, X):
        """
        Fit the model for standard deviation estimation.

        :param X: DataFrame with the timeseries data
        :return: Fitted model
        """
        X = pd.DataFrame(X)  # Ensure that X is a DataFrame
        if self.by_instant:
            # Fit by "instant"
            return self._fit_by_instant(X)
        else:
            # Use traditional fitting approach
            y, X = dmatrices(self.formula, data=X, return_type="dataframe")
            self.model.fit(X, y)
            return self.model

    def _fit_by_instant(self, X):
        """
        Fit the model by "instant", assuming a time granularity.
        """
        X_grouped = X.groupby("instant").apply(self._fit_and_extract_coefs)
        return X_grouped

    def predict_fun(self, X):
        """
        Predict the model for standard deviation estimation.

        :param X: DataFrame with timeseries data
        :return: Predictions (as a list or array)
        """
        if self.by_instant:
            return self.model.predict(X)
        else:
            return self.model.predict(X).tolist()
