from corrclim.smoother import MultiSmoother, Smoother


class TimeseriesModel:
    def __init__(
        self,
        formula,
        by_instant: bool = False,
        granularity: str = None,
        smoothers: Smoother | MultiSmoother = None,
        **kwargs,
    ):
        self.formula = Formula(formula)
        self.model = None
        self.smoothers = smoothers
        self.by_instant = {"activate": by_instant, "granularity": granularity}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def check_timeseries(self, X, is_fitting=True):
        if not isinstance(X, TimeseriesDT):
            X = TimeseriesDT(X)

        missing_vars = self._get_missing_vars(X, is_fitting)

        if missing_vars:
            if "instant" in missing_vars and self.by_instant["activate"]:
                X.compute_instant(granularity=self.by_instant["granularity"])

            if any("shifted" in var for var in missing_vars):
                granularity = X.get_granularity(unit="hour")
                if is_fitting:
                    X.shift(
                        self.formula.get_all_variables_formula_base(), n=self.n_shift / granularity
                    )
            else:
                X.add_calendar()

        missing_vars = self._get_missing_vars(X, is_fitting)
        if missing_vars:
            if not any("shifted" in var for var in missing_vars):
                raise ValueError(
                    f"Variables in formula not found in the dataset: {', '.join(missing_vars)}"
                )
        return X

    def fit(self, outputs, inputs):
        print(f"Fitting the model {type(self).__name__} ...")

        outputs = TimeseriesDT(outputs, is_output=True)
        inputs = TimeseriesDT(inputs)

        X = outputs.merge(inputs)
        X = self.check_timeseries(X, is_fitting=True)

        if self.smoothers:
            X = self.smoothers.fit_smooth(X)

        self.model = self.fit_fun(self.model, X)
        self._set_status(1)

        print("Model fitted!")

    def predict(self, X):
        if self._status < 1:
            raise ValueError("Please fit the model first using the fit() method.")
        print(f"Predicting using the model {type(self).__name__} ...")

        X = TimeseriesDT(X)
        X = self.check_timeseries(X, is_fitting=False)

        if self.smoothers:
            X = self.smoothers.smooth(X)

        return self.predict_fun(self.model, X)

    def export(self, path):
        if not path.lower().endswith(".pkl"):
            raise ValueError("File path should have extension .pkl")
        with open(path, "wb") as f:
            import pickle

            pickle.dump(self, f)

    def _set_status(self, value):
        if value not in {0, 1, 2}:
            raise ValueError("Status must be 0, 1, or 2.")
        self._status = value

    def _get_missing_vars(self, X, is_fitting):
        if is_fitting:
            missing_vars = set(self.formula.get_all_variables()) - set(X.get_variables_name())
            if self.by_instant["activate"]:
                missing_vars.add("instant")
        else:
            missing_vars = set(self.formula.get_explanatory_variables()) - set(
                X.get_variables_name()
            )
            if self.by_instant["activate"]:
                missing_vars.add("instant")
        return missing_vars
