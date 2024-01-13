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
