import pandas as pd
from pychronoboost.impute.timestep_impute import get_timestep_imputation_strategy
from pychronoboost.impute.value_impute import (
    get_value_imputation_strategy,
)
from pychronoboost.timeseries.feature_generator import TimeSeriesFeatureGenerator
from pychronoboost.timeseries.feature_selector import get_feature_selector


class TimeSeriesData:
    def __init__(self, data, timestep_column):
        self.data = data
        self.timestep_column = timestep_column
        self._validate_data()
        self.original_feature_columns = self.data.columns.tolist()
        

    def _validate_data(self):
        """
        Validates the input data to ensure it meets the requirements.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        if self.timestep_column not in self.data.columns:
            raise ValueError(
                f"The timestep column '{self.timestep_column}' is not in the DataFrame."
            )

    def process_timeseries_features(
        self,
        feature_columns,
        target_column,
        value_impute_strategy="last",
        max_window_size=3,
        feature_selector_model="XGB",
        max_features=5,
    ):
        self.impute_timesteps()
        self.impute_values(feature_columns, value_impute_strategy)
        generated_features = self.generate_features(feature_columns, max_window_size)
        self.impute_values(generated_features, "last")
        self.select_features(
            generated_features, target_column, max_features, feature_selector_model
        )
        return self.data

    def impute_timesteps(self):
        """
        Impute missing timesteps in the time series data.
        """
        timestep_strategy = get_timestep_imputation_strategy(
            self.data, self.timestep_column
        )
        imputed_timesteps = timestep_strategy.impute(
            self.data[[self.timestep_column]], self.timestep_column
        )

        # Merge the original data with the imputed timesteps
        self.data = pd.merge(
            imputed_timesteps, self.data, on=self.timestep_column, how="left"
        )

    def impute_values(self, value_columns, strategy):
        imputer = get_value_imputation_strategy(strategy)

        for col in value_columns:
            if col not in self.data.columns:
                raise ValueError(
                    f"column {col} not in input dataframe columns {self.data.columns}"
                )
            self.data[col] = imputer.impute(self.data[col])

    def generate_features(self, columns, max_window_size):
        feature_generator = TimeSeriesFeatureGenerator(max_window_size)
        all_generated_features = []
        for col in columns:
            generated_features = feature_generator.generate_features(self.data, col)
            all_generated_features += generated_features

        return all_generated_features

    def select_features(
        self, feature_columns, target_column, max_features=5, selector_model="XGB"
    ):
        feature_selector = get_feature_selector(selector_model, max_features)
        feature_selector.select_features(
            self.data,
            feature_columns,
            target_column,
            self.timestep_column,
            self.original_feature_columns,
        )
