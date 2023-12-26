import pandas as pd
from pychronoboost.pychronoboost.utils.value_impute import (
    LastValueImputation,
    ZeroImputation,
    LinearImputation,
)
from pychronoboost.pychronoboost.utils.timestep_impute import get_timestep_imputation_strategy
from pychronoboost.pychronoboost.feature_generator.timeseries_feature_generator import TimeSeriesFeatureGenerator

class TimeSeriesData:
    value_imputation_strategy = {
        "last": LastValueImputation(),
        "zero": ZeroImputation(),
        "linear": LinearImputation(),
    }

    def __init__(
        self,
        data: pd.DataFrame,
        timestep_column: str,
        value_column: str,
        impute_missing_steps: str = "last",
        max_window_size: int = 10,
        preprocess: bool = False,
    ):
        """
        Initialize the TimeSeriesData object.

        :param data: The time series data, expected as a Pandas DataFrame.
        :param timestep_column: The name of the column containing the timestep information.
        :param impute_missing_steps: The strategy for value imputation ('last', 'zero', 'linear').
        :param max_window_size: The maximum window size for feature generation.
        """
        self.data = data
        self.timestep_column = timestep_column
        self.value_column = value_column
        self.impute_missing_steps = impute_missing_steps
        self.max_window_size = max_window_size
        self.validate_data()
        if preprocess:
            self._preprocess()

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

        if self.value_column not in self.data.columns:
            raise ValueError(
                f"The value column '{self.value_column}' is not in the DataFrame."
            )

    def _preprocess(self):
        """
        Preprocess the data by sorting, imputing missing timesteps, and then imputing missing values.
        """
        self.data.sort_values(by=self.timestep_column, inplace=True)
        self._impute_timesteps()
        self._impute_values()

    def _impute_timesteps(self):
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

    def _impute_values(self):
        """
        Impute missing values in the time series data using the specified strategy.
        """
        if self.impute_missing_steps not in self.value_imputation_strategy:
            raise ValueError(
                "Invalid value imputation strategy. Choose 'last', 'zero', or 'linear'."
            )

        # Adjusted to pass only the data to the impute method
        value_impute_method = self.value_imputation_strategy[self.impute_missing_steps]
        self.data[self.value_column] = value_impute_method.impute(self.data[self.value_column])


# Example usage:
# ts_data = TimeSeriesData(your_dataframe, 'timestamp', 'last', 10)
# ts_data._preprocess()
