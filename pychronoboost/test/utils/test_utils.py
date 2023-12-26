import pandas as pd
import pytest
from pychronoboost.src.utils.helpers import (
    check_timeseries_type,
    TIMESTEP_INTEGER,
    TIMESTEP_FLOAT,
    TIMESTEP_DATE,
    TIMESTEP_DATETIME,
)


def test_check_timeseries_type_integer():
    df = pd.DataFrame({"time": [1, 2, 3, 4]})
    assert check_timeseries_type(df, "time") == TIMESTEP_INTEGER


def test_check_timeseries_type_float():
    df = pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0]})
    assert check_timeseries_type(df, "time") == TIMESTEP_FLOAT


def test_check_timeseries_type_date():
    dates = [
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-02").date(),
        pd.Timestamp("2020-01-03").date(),
    ]
    df = pd.DataFrame({"time": dates})
    assert check_timeseries_type(df, "time") == TIMESTEP_DATE


def test_check_timeseries_type_datetime():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2020-01-01 12:00", "2020-01-02 13:00", "2020-01-03 14:00"]
            )
        }
    )
    assert check_timeseries_type(df, "time") == TIMESTEP_DATETIME


def test_check_timeseries_type_column_not_found():
    df = pd.DataFrame({"time": [1, 2, 3]})
    with pytest.raises(ValueError) as excinfo:
        check_timeseries_type(df, "not_found_column")
    assert "The column 'not_found_column' is not in the DataFrame." in str(
        excinfo.value
    )


def test_check_timeseries_type_invalid_data():
    df = pd.DataFrame({"time": ["not a date", "also not a date"]})
    with pytest.raises(ValueError) as excinfo:
        check_timeseries_type(df, "time")
    assert (
        "The column must contain integers, floats, or be convertible to dates/datetime."
        in str(excinfo.value)
    )
