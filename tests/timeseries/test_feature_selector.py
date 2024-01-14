import pytest
import pandas as pd
from xgboost import XGBRegressor
from pychronoboost.timeseries.feature_selector import (
    XGBoostFeatureSelector,
)


@pytest.fixture
def sample_time_series_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
            "feature1": [5, 4, 3, 2, 1],
            "feature2": [1, 2, 1, 2, 1],
        }
    )


def test_xgboost_feature_selector(sample_time_series_data):
    selector = XGBoostFeatureSelector(num_features=1)
    selector.select_features(
        sample_time_series_data, ['feature1','feature2'],"value", "timestamp"
    )

    # Check if the selector returns a DataFrame
    assert isinstance(sample_time_series_data, pd.DataFrame)

    # Check if the timestamp and value columns are preserved
    assert "timestamp" in sample_time_series_data.columns
    assert "value" in sample_time_series_data.columns

    # Check if only one additional feature is selected
    assert len(sample_time_series_data.columns) == 3


@pytest.mark.parametrize("num_features", [1, 2])
def test_xgboost_feature_selection_num_features(sample_time_series_data, num_features):
    selector = XGBoostFeatureSelector(num_features=num_features)
    selector.select_features(
        sample_time_series_data, ['feature1','feature2'],"value", "timestamp"
    )

    # Check if the number of features selected is correct
    assert (
        len(sample_time_series_data.drop(["timestamp", "value"], axis=1).columns) == num_features
    )


def test_xgboost_feature_importance(sample_time_series_data):
    # This test assumes feature1 is more important based on its inverse relationship with 'value'
    selector = XGBoostFeatureSelector(num_features=1)
    selector.select_features(
        sample_time_series_data, ['feature1','feature2'],"value", "timestamp"
    )

    # Check if feature1 is selected as it's expected to be the most important
    assert "feature1" in sample_time_series_data.columns
