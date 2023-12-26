from abc import ABC, abstractmethod
import pandas as pd
from xgboost import XGBRegressor


class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def select_features(self, data: pd.DataFrame, target_column: str):
        """
        Select important features from the data.

        :param data: The DataFrame containing features and target.
        :param target_column: The name of the target column.
        :return: DataFrame with selected features.
        """
        pass


class XGBoostFeatureSelector(FeatureSelectionStrategy):
    def __init__(self, num_features: int):
        """
        Initialize XGBoostFeatureSelector.

        :param num_features: Number of top features to select.
        """
        self.num_features = num_features

    def select_features(
        self, data: pd.DataFrame, target_column: str, timestamp_column: str
    ):
        """
        Select important features from the data using XGBoost.

        :param data: The DataFrame containing features and target.
        :param target_column: The name of the target column.
        :param timestamp_column: The name of the timestamp column.
        :return: DataFrame with selected features, including timestamp and target columns.
        """
        X = data.drop([target_column, timestamp_column], axis=1)
        y = data[target_column]

        model = XGBRegressor()
        model.fit(X, y)

        # Get feature importances and select top features
        importance = pd.Series(model.feature_importances_, index=X.columns)
        selected_features = importance.nlargest(self.num_features).index.tolist()

        # Include the timestamp column and target column in the final DataFrame
        return data[[timestamp_column] + selected_features + [target_column]]


# Example usage
# feature_selector = XGBoostFeatureSelector(num_features=10)
# selected_data = feature_selector.select_features(ts_data.data, ts_data.value_column, ts_data.timestep_column)
