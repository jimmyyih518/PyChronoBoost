import pandas as pd
from pychronoboost.src.utils.value_impute import (
    LastValueImputation,
    ZeroImputation,
    LinearImputation,
)
 

class TimeSeriesData:
    imputation_strategy = {
        "last": LastValueImputation(),
        "zero": ZeroImputation(),
        "linear": LinearImputation(),
    }

    def __init__(
        self,
        data: pd.DataFrame,
        timestep_column: str,
        impute_missing_steps: str = "last",
        max_window_size: int = 10,
    ):
        """
        Initialize the TimeSeriesData object.

        :param data: The time series data, expected as a Pandas DataFrame.
        :param timestep_column: The name of the column containing the timestep information.
        :param impute_missing_steps: The strategy for imputation ('last', 'zero', 'linear').
        :param max_window_size: The maximum window size for feature generation.
        """
        self.data = data
        self.timestep_column = timestep_column
        self.impute_missing_steps = impute_missing_steps
        self.max_window_size = max_window_size
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

    def _preprocess(self):
        """
        Preprocess the data by sorting and imputing missing values.
        """
        self.data.sort_values(by=self.timestep_column, inplace=True)
        self._impute()

    def _impute(self):
        """
        Impute missing dates and values in the time series data using the specified strategy.
        """
        if self.impute_missing_steps not in self.imputation_strategy:
            raise ValueError(
                "Invalid imputation strategy. Choose 'last', 'zero', or 'linear'."
            )

        impute_method = self.imputation_strategy[self.impute_missing_steps]
        self.data = impute_method.impute(self.data, self.timestep_column)
