import pandas as pd
from typing import List
from pychronoboost.impute.timestep_impute import get_timestep_imputation_strategy
from pychronoboost.impute.value_impute import get_value_imputation_strategy
from pychronoboost.timeseries.feature_generator import TimeSeriesFeatureGenerator
from pychronoboost.timeseries.feature_selector import get_feature_selector


class TimeSeriesData:
    def __init__(self, data: pd.DataFrame, timestep_column: str):
        """
        Initializes the TimeSeriesData object.

        Args:
            data (pd.DataFrame): The pandas DataFrame containing the time series data.
            timestep_column (str): The name of the column in 'data' that represents the timestep.

        Raises:
            ValueError: If 'data' is not a pandas DataFrame or if 'timestep_column' is not in 'data'.
        """
        self.data = data
        self.timestep_column = timestep_column
        self._validate_data()
        self.original_feature_columns = self.data.columns.tolist()

    def _validate_data(self) -> None:
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
        feature_columns: List[str],
        target_column: str,
        value_impute_strategy: str = "last",
        max_window_size: int = 3,
        feature_selector_model: str = "XGB",
        max_features: int = 5,
    ) -> pd.DataFrame:
        """
        Processes time series features including imputation and feature generation.

        Args:
            feature_columns (List[str]): A list of column names to be used for feature generation.
            target_column (str): The name of the target column.
            value_impute_strategy (str): Strategy for imputing missing values.
            max_window_size (int): Maximum window size for feature generation.
            feature_selector_model (str): Model to use for feature selection.
            max_features (int): Maximum number of features to select.

        Returns:
            pd.DataFrame: The processed DataFrame with imputed and selected features.
        """
        self.impute_timesteps()
        self.impute_values(feature_columns, value_impute_strategy)
        generated_features = self.generate_features(feature_columns, max_window_size)
        self.impute_values(generated_features, "last")
        self.select_features(
            generated_features, target_column, max_features, feature_selector_model
        )
        return self.data

    def impute_timesteps(self) -> None:
        """
        Imputes missing timesteps in the time series data.
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

    def impute_values(self, value_columns: List[str], strategy: str) -> None:
        """
        Imputes missing values in specified columns of the DataFrame.

        Args:
            value_columns (List[str]): List of column names for which to impute missing values.
            strategy (str): Strategy to use for value imputation.

        Raises:
            ValueError: If any of the specified columns are not in the DataFrame.
        """
        imputer = get_value_imputation_strategy(strategy)

        for col in value_columns:
            if col not in self.data.columns:
                raise ValueError(
                    f"column {col} not in input dataframe columns {self.data.columns}"
                )
            self.data[col] = imputer.impute(self.data[col])

    def generate_features(self, columns: List[str], max_window_size: int) -> List[str]:
        """
        Generates new features based on specified columns and window size.

        Args:
            columns (List[str]): List of column names to use for feature generation.
            max_window_size (int): Maximum window size for generating features.

        Returns:
            List[str]: A list of names of the generated features.
        """
        feature_generator = TimeSeriesFeatureGenerator(max_window_size)
        all_generated_features = []
        for col in columns:
            generated_features = feature_generator.generate_features(self.data, col)
            all_generated_features += generated_features

        return all_generated_features

    def select_features(
        self,
        feature_columns: List[str],
        target_column: str,
        max_features: int = 5,
        selector_model: str = "XGB",
    ) -> None:
        """
        Selects the most relevant features based on the specified selection model.

        Args:
            feature_columns (List[str]): List of column names to consider for selection.
            target_column (str): The name of the target column.
            max_features (int): Maximum number of features to select.
            selector_model (str): Model to use for feature selection.
        """
        feature_selector = get_feature_selector(selector_model, max_features)
        feature_selector.select_features(
            self.data,
            feature_columns,
            target_column,
            self.timestep_column,
            self.original_feature_columns,
        )
