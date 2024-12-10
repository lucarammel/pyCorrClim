from dataclasses import dataclass
from typing import Optional

from loguru import logger

from corrclim.operator import Operator, OperatorAdditive
from corrclim.timeseries_dt import TimeseriesDT
from corrclim.timeseries_model.timeseries_model import TimeseriesModel
from corrclim.timeseries_std_model import TimeseriesStdModel


@dataclass
class ClimaticCorrector:
    timeseries_model: TimeseriesModel
    timeseries_std_model: Optional[TimeseriesStdModel] = None
    operator: Operator = OperatorAdditive()

    def __post_init__(self):
        if not isinstance(self.timeseries_model, TimeseriesModel):
            raise ValueError(
                "Climatic model given is not of type TimeseriesModel. Please provide a valid timeseries_model."
            )

        if self.timeseries_std_model:
            if not isinstance(self.timeseries_std_model, TimeseriesStdModel):
                raise ValueError(
                    "The standard deviation model given is not of type TimeseriesStdModel."
                )
            if not isinstance(self.operator, OperatorAdditive):
                raise ValueError(
                    "Please provide an Operator2Moments with a standard deviation model."
                )

    def fit(self, timeseries, weather_observed, fold_varname=None):
        timeseries = TimeseriesDT(timeseries, is_output=True)
        weather_observed = TimeseriesDT(weather_observed)

        if isinstance(self.operator, OperatorAdditive) and self.timeseries_std_model:
            self.timeseries_std_model.fit(timeseries, weather_observed, fold_varname)
        self.timeseries_model.fit(timeseries, weather_observed)

    def apply(self, timeseries, weather_observed, weather_target):
        logger.info("Applying the Climate Correction...")

        timeseries = TimeseriesDT(timeseries, is_output=True)
        weather_observed = TimeseriesDT(weather_observed)
        weather_target = TimeseriesDT(weather_target)

        weather_target, weather_observed = timeseries.align(weather_target, weather_observed)

        logger.info("Prediction on the target:")
        y_pred_target = self.timeseries_model.predict(weather_target)

        logger.info("Prediction on the observed:")
        y_pred_observed = self.timeseries_model.predict(weather_observed)

        if isinstance(self.operator, OperatorAdditive) and self.timeseries_std_model:
            logger.info("Prediction on the target standard deviation:")
            y_std_target = self.timeseries_std_model.predict(weather_target)

            logger.info("Prediction on the observed standard deviation:")
            y_std_observed = self.timeseries_std_model.predict(weather_observed)

            y_climate_corrected = self.operator.apply(
                timeseries=timeseries,
                y_pred_observed=y_pred_observed,
                y_pred_target=y_pred_target,
                y_std_observed=y_std_observed,
                y_std_target=y_std_target,
            )
        else:
            y_climate_corrected = self.operator.apply(
                timeseries=timeseries, y_pred_observed=y_pred_observed, y_pred_target=y_pred_target
            )

        logger.info("Climate correction ended.")
        return y_climate_corrected

    def get_operator(self):
        return self.operator

    def export(self, path: str):
        if not path.lower().endswith(".pkl"):
            raise ValueError("File path should have extension .pkl")

        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)
