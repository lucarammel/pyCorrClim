import pandas as pd


class TimeseriesDT:
    def __init__(
        self, timeseries, is_output=False, format_date="%Y-%m-%d %H:%M:%S", timezone="UTC"
    ):
        self.format_date = format_date
        self.timezone = timezone
        self.timeseries = None

        if isinstance(timeseries, TimeseriesDT):
            # Handle initialization from an existing TimeseriesDT object
            self.timeseries = timeseries.get_timeseries()
            self.format_date = timeseries.format_date
            self.timezone = timeseries.timezone
        else:
            try:
                self.timeseries = pd.DataFrame(timeseries)
            except Exception as e:
                raise ValueError("Timeseries cannot be formatted as a DataFrame") from e

            self.timeseries = self.rename_time_column(self.timeseries)
            try:
                self.timeseries["time"] = (
                    pd.to_datetime(self.timeseries["time"])
                    .dt.tz_localize(self.timezone)
                    .dt.strftime(self.format_date)
                )
                self.timeseries["time"] = pd.to_datetime(
                    self.timeseries["time"], format=self.format_date
                )
            except Exception as e:
                raise ValueError("Invalid 'time' column in TimeseriesDT") from e

        if is_output:
            if len(self.timeseries.columns) > 2:
                raise ValueError("If output, timeseries should have 2 columns")
            value_column = [col for col in self.timeseries.columns if col.lower() != "time"]
            self.timeseries.rename(columns={value_column[0]: "y"}, inplace=True)

    def rename_time_column(self, df):
        time_patterns = ["TIME", "DATE"]
        for col in df.columns:
            if col.upper() in time_patterns:
                df.rename(columns={col: "time"}, inplace=True)
                break
        return df

    def remove_na(self, inplace=True):
        dt = self.timeseries.dropna()
        if inplace:
            self.timeseries = dt
            return self
        else:
            return TimeseriesDT(dt)

    def set_timeseries(self, timeseries):
        self.timeseries = timeseries

    def get_timeseries(self):
        return self.timeseries.copy()

    def set_format_date(self, format_date=None):
        if format_date:
            self.format_date = format_date
        self.timeseries["time"] = (
            pd.to_datetime(self.timeseries["time"])
            .dt.tz_localize(self.timezone)
            .dt.strftime(self.format_date)
        )
        self.timeseries["time"] = pd.to_datetime(self.timeseries["time"], format=self.format_date)

    def set_timezone(self, timezone):
        self.timezone = timezone
        self.timeseries["time"] = pd.to_datetime(self.timeseries["time"]).dt.tz_localize(
            self.timezone
        )

    def sort(self, variable, inplace=True):
        sorted_df = self.timeseries.sort_values(by=variable)
        if inplace:
            self.timeseries = sorted_df
        else:
            return TimeseriesDT(sorted_df)

    def compute_period_start(self, granularity, inplace=True):
        timeseries = self.timeseries.copy()

        if granularity == "hour":
            timeseries["period_start"] = timeseries["time"].dt.floor("H")
        elif granularity == "day":
            timeseries["period_start"] = timeseries["time"].dt.date
        elif granularity == "week":
            timeseries["period_start"] = (
                timeseries["time"].dt.to_period("W").apply(lambda r: r.start_time)
            )
        elif granularity == "month":
            timeseries["period_start"] = (
                timeseries["time"].dt.to_period("M").apply(lambda r: r.start_time)
            )
        elif granularity == "year":
            timeseries["period_start"] = (
                timeseries["time"].dt.to_period("Y").apply(lambda r: r.start_time)
            )
        else:
            raise ValueError("Unsupported granularity")

        if inplace:
            self.timeseries = timeseries
        else:
            return TimeseriesDT(timeseries)

    def aggregate(self, granularity, func=np.mean, inplace=True):
        self.compute_period_start(granularity, inplace=True)
        aggregated = self.timeseries.groupby("period_start").agg(func).reset_index()
        aggregated.rename(columns={"period_start": "time"}, inplace=True)

        if inplace:
            self.timeseries = aggregated
        else:
            return TimeseriesDT(aggregated)

    def groupby(self, granularity, func=np.mean):
        timeseries = self.timeseries.copy()

        if granularity == "hour":
            timeseries["time"] = timeseries["time"].dt.hour
        elif granularity == "wday":
            timeseries["time"] = timeseries["time"].dt.weekday
        elif granularity == "week":
            timeseries["time"] = timeseries["time"].dt.isocalendar().week
        elif granularity == "month":
            timeseries["time"] = timeseries["time"].dt.month
        elif granularity == "year":
            timeseries["time"] = timeseries["time"].dt.year
        else:
            raise ValueError("Unsupported granularity")

        return timeseries.groupby("time").agg(func).reset_index()

    def select(self, variables, inplace=True):
        selected = self.timeseries[["time"] + variables]
        if inplace:
            self.timeseries = selected
        else:
            return TimeseriesDT(selected)

    def remove_duplicated(self, variables=["time"], inplace=True):
        deduplicated = self.timeseries.drop_duplicates(subset=variables)
        if inplace:
            self.timeseries = deduplicated
        else:
            return TimeseriesDT(deduplicated)

    def assign(self, name, vector, inplace=True):
        self.timeseries[name] = vector
        if inplace:
            pass
        else:
            return TimeseriesDT(self.timeseries)

    def merge(self, other, by="time", how="inner", suffixes=(".x", ".y"), inplace=True):
        if isinstance(other, TimeseriesDT):
            other = other.timeseries
        merged = pd.merge(self.timeseries, other, on=by, how=how, suffixes=suffixes)

        if inplace:
            self.timeseries = merged
        else:
            return TimeseriesDT(merged)

    def remove_variables(self, variables, inplace=True):
        filtered = self.timeseries.drop(columns=variables)
        if inplace:
            self.timeseries = filtered
        else:
            return TimeseriesDT(filtered)

    def compute_degree_days(
        self,
        temperature_column,
        all=True,
        cooling=False,
        threshold_cooling=18,
        threshold_heating=15,
        inplace=True,
    ):
        timeseries = self.timeseries.copy()
        if all:
            timeseries["HDD"] = np.maximum(0, threshold_heating - timeseries[temperature_column])
            timeseries["CDD"] = np.maximum(0, timeseries[temperature_column] - threshold_cooling)
        elif cooling:
            timeseries["CDD"] = np.maximum(0, timeseries[temperature_column] - threshold_cooling)
        else:
            timeseries["HDD"] = np.maximum(0, threshold_heating - timeseries[temperature_column])

        if inplace:
            self.timeseries = timeseries
        else:
            return TimeseriesDT(timeseries)

    def get_granularity(self, unit="hour"):
        delta = self.timeseries["time"].diff().iloc[1]
        if unit == "hour":
            return delta.total_seconds() / 3600
        elif unit == "minute":
            return delta.total_seconds() / 60
        elif unit == "second":
            return delta.total_seconds()
        else:
            raise ValueError("Unsupported unit")

    def rename(self, old_cols, new_cols, inplace=True):
        renamed = self.timeseries.rename(columns=dict(zip(old_cols, new_cols)))
        if inplace:
            self.timeseries = renamed
        else:
            return TimeseriesDT(renamed)

    def filter_dataset(
        self,
        y_shifted,
        var,
        var_shifted,
        threshold,
        q_max=0.8,
        q_min=0.2,
        IC_width=1.5,
        inferior=True,
        inplace=True,
    ):
        timeseries = self.timeseries.copy()

        # Filter non-NA values
        filtered = timeseries.dropna(subset=[y_shifted, var])

        # Apply threshold conditions
        if inferior:
            filtered = filtered[
                (filtered[var] <= threshold)
                & ((filtered[var] - filtered[var_shifted]) <= threshold)
            ]
        else:
            filtered = filtered[
                (filtered[var] >= threshold)
                & ((filtered[var] - filtered[var_shifted]) >= threshold)
            ]

        # Apply quantile and confidence interval filtering
        y_shifted_min = filtered[y_shifted].quantile(q_min)
        y_shifted_max = filtered[y_shifted].quantile(q_max)
        range_width = IC_width * (y_shifted_max - y_shifted_min)

        filtered = filtered[
            (filtered[y_shifted] > y_shifted_min - range_width)
            & (filtered[y_shifted] < y_shifted_max + range_width)
        ]

        if inplace:
            self.timeseries = filtered
            return self
        else:
            return TimeseriesDT(filtered)

    def export(self, path, as_data_table=True, file_format="csv"):
        file_format = file_format.lower()

        # Validate format and file extension
        if not path.lower().endswith(file_format):
            raise ValueError(
                "Format and file extension don't match. Please fix the path or the format."
            )

        if as_data_table:
            if file_format == "csv":
                self.timeseries.to_csv(path, index=False)
            elif file_format == "rds":
                import pyreadr

                pyreadr.write_rds(path, {"timeseries": self.timeseries})
            else:
                raise ValueError("Format not supported. Please use 'csv' or 'rds'.")
        else:
            # Export the whole object using pickle
            import pickle

            with open(path, "wb") as f:
                pickle.dump(self, f)

    def _rename_time_column(self):
        if len(self.timeseries.columns) < 2:
            raise ValueError("Invalid data. Please provide at least two columns.")

        time_column_candidates = [col for col in self.timeseries.columns if "time" in col.lower()]

        if len(time_column_candidates) == 1:
            self.timeseries.rename(columns={time_column_candidates[0]: "time"}, inplace=True)
        elif len(time_column_candidates) > 1:
            # Do nothing
            pass
        else:
            raise ValueError("Please provide a column with a time pattern.")

    def add_suffix(self, variables, suffix, inplace=True):
        """
        Add a suffix to the specified variables' names in the timeseries dataset.

        Parameters:
        - variables (list of str): The variables (columns) to which the suffix will be added.
        - suffix (str): The suffix to add to the column names.
        - inplace (bool): If True, modify the current instance. Otherwise, return a new instance.

        Returns:
        - TimeseriesDT (optional): A new instance with modified column names if `inplace=False`.
        """
        timeseries = self.timeseries.copy()
        timeseries.rename(columns={var: f"{var}{suffix}" for var in variables}, inplace=True)

        if inplace:
            self.timeseries = timeseries
            return self
        else:
            return TimeseriesDT(timeseries)
