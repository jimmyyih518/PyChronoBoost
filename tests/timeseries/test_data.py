import pandas as pd
import pytest
from pychronoboost.timeseries.data import TimeSeriesData


def test_time_series_data_importable():
    try:
        from pychronoboost.timeseries.data import TimeSeriesData
    except ImportError as e:
        assert False, f"Failed to import TimeSeriesData, {e}"


# Fixture for sample data
@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "time": pd.date_range(start="2021-01-01", periods=4, freq="D"),
            "value": [1, None, 3, 4],
        }
    )


# Test Initialization and Data Validation
def test_initialization(sample_data):
    ts_data = TimeSeriesData(sample_data, "time")
    assert ts_data.timestep_column == "time"
    assert ts_data.data.equals(sample_data)


def test_initialization_with_invalid_data():
    with pytest.raises(ValueError):
        TimeSeriesData("not a dataframe", "time")


def test_initialization_with_invalid_column(sample_data):
    with pytest.raises(ValueError):
        TimeSeriesData(sample_data, "nonexistent_column")


# Test Impute Values Method
def test_impute_values(sample_data):
    ts_data = TimeSeriesData(sample_data, "time")
    ts_data.impute_values(["value"], "linear")  # Assuming 'linear' is a valid strategy
    assert (
        ts_data.data["value"].isnull().sum() == 0
    )  # Assuming linear strategy fills all NaNs


def test_impute_values_with_invalid_column(sample_data):
    ts_data = TimeSeriesData(sample_data, "time")
    with pytest.raises(ValueError):
        ts_data.impute_values(["nonexistent_column"], "linear")


def test_process_timeseries_features():
    # Mock data and timestep column
    data = pd.DataFrame(
        {
            "timestamp": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-04",
                "2022-01-05",
                "2022-01-06",
            ],
            "value1": [1, None, 4, 5, None],
            "value2": [4, 2, 8, 5, 3],
            "target": [1, 2, 3, 4, 5],
        }
    )

    ts_data = TimeSeriesData(data, "timestamp")
    processed_data = ts_data.process_timeseries_features(
        ["value1", "value2"], "target", max_features=3
    )

    assert isinstance(processed_data, pd.DataFrame)
    processed_data_columns = processed_data.columns.tolist()
    assert len(processed_data_columns) == 7
    assert "timestamp" in processed_data_columns
    assert "value1" in processed_data_columns
    assert "value2" in processed_data_columns
    assert "target" in processed_data_columns
    assert len(processed_data) == 6
