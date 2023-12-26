import pytest
import pandas as pd
from pychronoboost.pychronoboost.data.timeseries_data import TimeSeriesData


def test_time_series_data_preprocess():
    # Mock data and timestep column
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=3, freq="D"),
            "value": [1, None, 3],  # Missing value on 2020-01-02
        }
    )
    timestep_column = "timestamp"
    value_column = "value"

    # Initialize TimeSeriesData
    ts_data = TimeSeriesData(
        data, timestep_column, value_column, impute_missing_steps="last"
    )

    # Before preprocessing
    assert ts_data.data.isnull().any().any()  # Check if there are any missing values

    # Preprocess
    ts_data._preprocess()

    # After preprocessing
    assert not ts_data.data.isnull().any().any()  # Ensure no missing values
    assert len(ts_data.data) == 3  # Check if length is still 3


def test_invalid_dataframe_input():
    with pytest.raises(ValueError):
        TimeSeriesData("not a dataframe", "timestamp", "value")


def test_invalid_column_input():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=3, freq="D"),
            "value": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError):
        TimeSeriesData(data, "nonexistent_column", "value")


# Additional tests can be added for specific imputation strategies
def test_time_series_data_preprocess_with_impute():
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
            "value": [1, None, 4, 5, None],
        }
    )
    timestep_column = "timestamp"
    value_column = "value"

    # Initialize TimeSeriesData
    ts_data = TimeSeriesData(
        data,
        timestep_column,
        value_column,
        impute_missing_steps="linear",
        preprocess=True,
    )

    expected = pd.DataFrame(
        {
            "timestamp": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
                "2022-01-06",
            ],
            "value": [1, 2, 3, 4, 5, 5],
        }
    )
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    pd.testing.assert_frame_equal(expected, ts_data.data, check_dtype=False)
