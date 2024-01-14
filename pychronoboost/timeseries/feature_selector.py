from abc import ABC, abstractmethod
import pandas as pd
from xgboost import XGBRegressor
from typing import List


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
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        timestamp_column: str,
        original_feature_columns: List[str] = [],
    ) -> None:
        """
        Select important features from the data using XGBoost.

        :param data: The DataFrame containing features and target.
        :param feature_columns: The names of all feature columns in a list
        :param target_column: The name of the target column.
        :param timestamp_column: The name of the timestamp column.
        :original_feature_columns: The names of the original feature columns in a list
        :return: None (Modifies dataframe in place)
        """
        model_data = data.dropna()
        X = model_data[feature_columns]
        y = model_data[target_column]

        model = XGBRegressor()
        model.fit(X, y)

        # Get feature importances and select top features
        importance = pd.Series(model.feature_importances_, index=X.columns)
        selected_features = importance.nlargest(self.num_features).index.tolist()

        # Include the timestamp column and target column in the final DataFrame
        columns_to_keep = (
            [timestamp_column]
            + selected_features
            + [target_column]
            + original_feature_columns
        )
        columns_to_drop = [col for col in data.columns if col not in columns_to_keep]

        data.drop(columns=columns_to_drop, inplace=True)


def get_feature_selector(
    selector_model: str, num_features: int
) -> FeatureSelectionStrategy:
    """
    Finds the corresponding the feature selection class method

    :param selector_model: name of the feature selector model
    :param num_features: Max number of features to be returned after feature selection
    :return: feature selection class corresponding to input selector_model
    """
    feature_selectors = {"XGB": XGBoostFeatureSelector(num_features)}
    if selector_model not in feature_selectors:
        raise NotImplementedError(f"Feature Selector {selector_model} not available")
    return feature_selectors[selector_model]


# Example usage
# feature_selector = XGBoostFeatureSelector(num_features=10)
# selected_data = feature_selector.select_features(ts_data.data, ts_data.value_column, ts_data.timestep_column)
