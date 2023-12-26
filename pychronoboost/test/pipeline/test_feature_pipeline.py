import pandas as pd
import pytest
from pychronoboost.pychronoboost.pipeline.feature_pipeline import FeaturePipeline


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


def test_feature_pipeline_execution(sample_time_series_data):
    pipeline = FeaturePipeline("timestamp", "value", max_window_size=3, num_features=2)
    processed_data = pipeline.execute(sample_time_series_data, preprocess=False)

    # Check if the pipeline returns a DataFrame
    assert isinstance(processed_data, pd.DataFrame)

    # Check if the timestamp and value columns are preserved
    assert "timestamp" in processed_data.columns
    assert "value" in processed_data.columns

    # Check if the correct number of features are selected (2 features + timestamp + value)
    assert len(processed_data.columns) == 4
