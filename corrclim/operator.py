from abc import ABC, abstractmethod

import numpy as np


# Abstract Base Operator
class Operator(ABC):
    """
    Base class for climate correction operators.
    """

    def apply(
        self, timeseries, y_pred_observed, y_pred_target, y_std_observed=None, y_std_target=None
    ):
        """
        Apply the climate correction operator.

        :param timeseries: (TimeseriesDT) The timeseries to apply climate correction on.
        :param y_pred_observed: (array-like) Prediction on observed weather.
        :param y_pred_target: (array-like) Prediction on target weather.
        :param y_std_observed: (array-like) Standard deviation on observed weather (optional).
        :param y_std_target: (array-like) Standard deviation on target weather (optional).
        :return: (TimeseriesDT) Climate-corrected timeseries.
        """
        print(f"Applying {self.__class__.__name__} for climate correction.")
        timeseries = TimeseriesDT(timeseries.get_timeseries())
        return self.apply_fun(
            timeseries, y_pred_observed, y_pred_target, y_std_observed, y_std_target
        )

    @abstractmethod
    def apply_fun(
        self, timeseries, y_pred_observed, y_pred_target, y_std_observed=None, y_std_target=None
    ):
        """
        Abstract method to implement the operator-specific logic.
        """
        pass


# OperatorTarget: Returns y_pred_target as the result
class OperatorTarget(Operator):
    def apply_fun(
        self, timeseries, y_pred_observed, y_pred_target, y_std_observed=None, y_std_target=None
    ):
        return TimeseriesDT(pd.DataFrame({"y_climate_corrected": y_pred_target}))


# OperatorAdditive: Adds delta (y_pred_target - y_pred_observed) to the timeseries
class OperatorAdditive(Operator):
    def apply_fun(
        self, timeseries, y_pred_observed, y_pred_target, y_std_observed=None, y_std_target=None
    ):
        timeseries_df = timeseries.get_timeseries()
        delta = y_pred_target - y_pred_observed
        timeseries_df["y_climate_corrected"] = timeseries_df["y"] + delta
        return TimeseriesDT(timeseries_df)


# OperatorMultiplicative: Multiplies values by the ratio of predictions
class OperatorMultiplicative(Operator):
    def apply_fun(
        self, timeseries, y_pred_observed, y_pred_target, y_std_observed=None, y_std_target=None
    ):
        timeseries_df = timeseries.get_timeseries()
        timeseries_df["y_climate_corrected"] = timeseries_df["y"] * (
            y_pred_target / y_pred_observed
        )
        return TimeseriesDT(timeseries_df)


# Operator2Moments: Preserves the first two distribution moments
class Operator2Moments(Operator):
    def apply_fun(self, timeseries, y_pred_observed, y_pred_target, y_std_observed, y_std_target):
        timeseries_df = timeseries.get_timeseries()

        # Handle cases where standard deviation is zero
        y_std_target = np.where(y_std_target == 0, 1, y_std_target)
        y_std_observed = np.where(y_std_observed == 0, 1, y_std_observed)

        timeseries_df["y_climate_corrected"] = y_pred_target + (y_std_target / y_std_observed) * (
            timeseries_df["y"] - y_pred_observed
        )
        return TimeseriesDT(timeseries_df)
