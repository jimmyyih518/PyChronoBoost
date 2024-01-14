import pytest
import pandas as pd
from pychronoboost.timeseries.feature_generator import (
    TimeSeriesFeatureGenerator,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
        }
    )


def test_generate_features_min(sample_data):
    feature_generator = TimeSeriesFeatureGenerator(max_window_size=3)
    feature_generator.generate_features(sample_data, "value")
    assert "value_min_3" in sample_data.columns
    assert sample_data["value_min_3"].iloc[2] == 1  # Min value in the first window of size 3


def test_generate_features_max(sample_data):
    feature_generator = TimeSeriesFeatureGenerator(max_window_size=3)
    feature_generator.generate_features(sample_data, "value")
    assert "value_max_3" in sample_data.columns
    assert sample_data["value_max_3"].iloc[2] == 3  # Max value in the first window of size 3


def test_generate_features_avg(sample_data):
    feature_generator = TimeSeriesFeatureGenerator(max_window_size=3)
    feature_generator.generate_features(sample_data, "value")
    assert "value_avg_3" in sample_data.columns
    assert (
        sample_data["value_avg_3"].iloc[2] == 2
    )  # Average value in the first window of size 3


def test_generate_features_nth(sample_data):
    feature_generator = TimeSeriesFeatureGenerator(max_window_size=3)
    feature_generator.generate_features(sample_data, "value")
    assert "value_nth_3" in sample_data.columns
    assert sample_data["value_nth_3"].iloc[2] == 1  # Nth (3rd) value from the start


def test_multiple_window_sizes(sample_data):
    feature_generator = TimeSeriesFeatureGenerator(max_window_size=3)
    feature_generator.generate_features(sample_data, "value")
    # Check if features for all window sizes (1 to 3) are generated
    for window_size in range(1, 4):
        assert f"value_min_{window_size}" in sample_data.columns
        assert f"value_max_{window_size}" in sample_data.columns
        assert f"value_avg_{window_size}" in sample_data.columns
        assert f"value_nth_{window_size}" in sample_data.columns
