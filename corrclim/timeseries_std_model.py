import numpy as np

from corrclim.timeseries_model import TimeseriesModel


class TimeseriesStdModel(TimeseriesModel):
    def __init__(
        self, formula, by_instant, granularity, conditional_expectation_model, *args, **kwargs
    ):
        """
        Initialize the TimeseriesStdModel class.

        :param formula: The formula object (e.g., value ~ temperature)
        :param by_instant: Additional parameter, needs to be defined in your implementation
        :param granularity: Granularity of the model
        :param conditional_expectation_model: A TimeseriesModel object to estimate the conditional expectation
        """
        super().__init__(formula, by_instant, granularity, *args, **kwargs)

        if not isinstance(conditional_expectation_model, TimeseriesModel):
            raise ValueError(
                "Please provide a TimeseriesModel object to estimate conditional expectation"
            )

        self.conditional_expectation_model = conditional_expectation_model

    def fit(self, outputs, inputs, fold_varname):
        """
        Fit the conditional variance model based on the conditional expectation one.

        :param outputs: The output/response data (pandas DataFrame or custom TimeseriesDT)
        :param inputs: The input data with data for both conditional expectation and conditional variance models
        :param fold_varname: The name of the variable in `inputs` to define CV folds
        """
        print("Fitting the TimeseriesStd Model.")

        # Assuming TimeseriesDT is a custom class, we will use pandas for this example
        if fold_varname not in inputs.columns:
            raise ValueError(
                "You need to provide the variable defining CV folds inside the input timeseries"
            )

        print("Performing a Cross Validation prediction with conditional expectation model")

        # Using the cv_predict method of the conditional expectation model
        output_cv_pred = self.conditional_expectation_model.cv_predict(
            outputs, inputs, fold_varname
        )
        output_cv_residual_sqrd = (outputs["y"] - output_cv_pred) ** 2

        outputs["y"] = output_cv_residual_sqrd

        print("Fitting now using the residuals squared")
        super().fit(outputs, inputs)

    def predict(self, inputs):
        """
        Predict the conditional standard deviation.

        :param inputs: The timeseries data to make predictions on (pandas DataFrame or custom TimeseriesDT)

        :return: The output timeseries as a vector from the model prediction
        """
        conditional_variance = super().predict(inputs)
        conditional_variance = np.maximum(0, conditional_variance)

        return np.sqrt(conditional_variance)
