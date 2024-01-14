import pandas as pd


class TimeSeriesFeatureGenerator:
    def __init__(self, max_window_size: int):
        """
        Initialize the FeatureGenerator object.

        :param max_window_size: The maximum window size for feature generation.
        """
        self.max_window_size = max_window_size
        self.generated_features = []

    def generate_features(self, data: pd.DataFrame, value_column: str):
        """
        Generate statistical features for the time series data.

        :param data: The time series data as a Pandas DataFrame.
        :param value_column: The name of the column containing the values.
        :return: DataFrame with generated features.
        """
        self.generated_features = []
        for window_size in range(1, self.max_window_size + 1):
            data = self._generate_window_features(data, value_column, window_size)

        return self.generated_features

    def _generate_window_features(
        self, data: pd.DataFrame, value_column: str, window_size: int
    ):
        """
        Generate features for a specific window size.

        :param data: The time series data as a Pandas DataFrame.
        :param value_column: The name of the column containing the values.
        :param window_size: The specific window size for feature generation.
        :return: DataFrame with features for the specific window size.
        """
        data[f"{value_column}_min_{window_size}"] = (
            data[value_column].rolling(window=window_size).min()
        )
        data[f"{value_column}_max_{window_size}"] = (
            data[value_column].rolling(window=window_size).max()
        )
        data[f"{value_column}_avg_{window_size}"] = (
            data[value_column].rolling(window=window_size).mean()
        )
        data[f"{value_column}_nth_{window_size}"] = data[value_column].shift(
            window_size - 1
        )

        self.generated_features += [
            f"{value_column}_min_{window_size}",
            f"{value_column}_max_{window_size}",
            f"{value_column}_avg_{window_size}",
            f"{value_column}_nth_{window_size}",
        ]

        return data


# Example usage in the TimeSeriesData class
# feature_generator = FeatureGenerator(max_window_size)
# data_with_features = feature_generator.generate_features(ts_data.data, ts_data.value_column)
