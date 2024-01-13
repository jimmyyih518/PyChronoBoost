import pandas as pd
from pychronoboost.impute.timestep_impute import get_timestep_imputation_strategy
from pychronoboost.impute.value_impute import (
    get_value_imputation_strategy,
)


class TimeSeriesData:
    def __init__(self, data, timestep_column):
        self.data = data
        self.timestep_column = timestep_column
        self.validate_data()

    def validate_data(self):
        """
        Validates the input data to ensure it meets the requirements.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        if self.timestep_column not in self.data.columns:
            raise ValueError(
                f"The timestep column '{self.timestep_column}' is not in the DataFrame."
            )

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
