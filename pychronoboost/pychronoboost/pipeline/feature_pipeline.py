import pandas as pd
from typing import List, Union
from pychronoboost.pychronoboost.data.timeseries_data import TimeSeriesData
from pychronoboost.pychronoboost.feature_generator.timeseries_feature_generator import (
    TimeSeriesFeatureGenerator,
)
from pychronoboost.pychronoboost.feature_generator.timeseries_feature_selector import (
    XGBoostFeatureSelector,
)
from pychronoboost.pychronoboost.output.output_formatter import OutputFormatter


class FeaturePipeline:
    def __init__(
        self,
        timestep_column: str,
        value_columns: Union[List[str], str],
        max_window_size: int,
        num_features: int,
    ):
        """
        Initialize the FeaturePipeline object.

        :param timestep_column: The name of the column containing the timestep information.
        :param value_columns: A list of names of the columns containing the values.
        :param max_window_size: The maximum window size for feature generation.
        :param num_features: Number of top features to select for each value column.
        """
        self.timestep_column = timestep_column
        self.max_window_size = max_window_size
        self.num_features = num_features

        if isinstance(value_columns, str):
            self.value_columns = [value_columns]
        else:
            self.value_columns = value_columns

    def execute(
        self,
        data: pd.DataFrame,
        preprocess: bool = False,
        save_to_csv: bool = False,
        csv_file_path: str = None,
    ):
        """
        Execute the feature pipeline process on the provided data.

        :param data: The time series data as a Pandas DataFrame.
        :param preprocess: Whether to preprocess the data.
        :param save_to_csv: Whether to save the output to a CSV file.
        :param csv_file_path: The file path for saving the output CSV. Required if save_to_csv is True.
        """
        processed_data = data.copy()

        for value_column in self.value_columns:
            # Initialize TimeSeriesData for each value column
            ts_data = TimeSeriesData(
                processed_data,
                self.timestep_column,
                value_column,
                preprocess=preprocess,
            )

            # Feature Generation
            feature_generator = TimeSeriesFeatureGenerator(self.max_window_size)
            ts_data.data = feature_generator.generate_features(
                ts_data.data, ts_data.value_column
            )

            # Feature Selection
            feature_selector = XGBoostFeatureSelector(self.num_features)
            ts_data.data = feature_selector.select_features(
                ts_data.data, ts_data.value_column, ts_data.timestep_column
            )
            # Drop the existing value column from the processed data to avoid duplication
            features_to_merge = ts_data.data.drop(
                columns=[self.timestep_column, value_column], errors="ignore"
            )

            # Merge the processed features back to the main DataFrame
            processed_data = pd.merge(
                processed_data, features_to_merge, on=self.timestep_column, how="left"
            )

        # Output Formatting
        output_formatter = OutputFormatter(processed_data)
        formatted_data = output_formatter.format_output()

        if save_to_csv:
            if csv_file_path is None:
                raise ValueError(
                    "CSV file path must be provided if save_to_csv is True."
                )
            output_formatter.save_to_csv(csv_file_path)

        return formatted_data


# Example usage
# pipeline = FeaturePipeline("timestamp", ["value1", "value2"], max_window_size=10, num_features=5)
# final_data = pipeline.execute(raw_data, preprocess=True, save_to_csv=True, csv_file_path="output.csv")
