import pytest
import pandas as pd
import numpy as np
from pychronoboost.impute.timestep_impute import (
    DateImputation,
    DateTimeImputation,
    IntegerImputation,
    FloatImputation,
    get_timestep_imputation_strategy,
)

from pychronoboost.utils import (
    check_timeseries_type,
    TIMESTEP_INTEGER,
    TIMESTEP_FLOAT,
    TIMESTEP_DATE,
    TIMESTEP_DATETIME,
)

# Sample data for testing
date_data = pd.DataFrame(
    {
        "date": pd.date_range("2020-01-01", periods=5, freq="2D"),
        "value": [1, 2, np.nan, 4, 5],
    }
)
datetime_data = pd.DataFrame(
    {
        "datetime": pd.date_range("2020-01-01", periods=5, freq="2H"),
        "value": [1, 2, np.nan, 4, 5],
    }
)
integer_data = pd.DataFrame({"integer": [1, 2, 4, 5], "value": [1, 2, 4, 5]})
float_data = pd.DataFrame({"float": [1.0, 2.0, 4.0, 5.0], "value": [1, 2, 4, 5]})


@pytest.mark.parametrize(
    "data,column,expected",
    [
        (date_data, "date", TIMESTEP_DATE),
        (datetime_data, "datetime", TIMESTEP_DATETIME),
        (integer_data, "integer", TIMESTEP_INTEGER),
        (float_data, "float", TIMESTEP_FLOAT),
    ],
)
def test_check_timeseries_type(data, column, expected):
    assert check_timeseries_type(data, column) == expected


def test_date_imputation():
    imputer = DateImputation()
    imputed_data = imputer.impute(date_data, "date")
    assert imputed_data["date"].is_monotonic_increasing


def test_datetime_imputation():
    imputer = DateTimeImputation()
    imputed_data = imputer.impute(datetime_data, "datetime")
    assert imputed_data["datetime"].is_monotonic_increasing


def test_integer_imputation():
    imputer = IntegerImputation()
    imputed_data = imputer.impute(integer_data, "integer")
    assert imputed_data["integer"].is_monotonic_increasing


def test_float_imputation():
    imputer = FloatImputation()
    imputed_data = imputer.impute(float_data, "float")
    assert imputed_data["float"].is_monotonic_increasing


def test_get_timestep_imputation_strategy():
    for data, column, strategy_class in [
        (date_data, "date", DateImputation),
        (datetime_data, "datetime", DateTimeImputation),
        (integer_data, "integer", IntegerImputation),
        (float_data, "float", FloatImputation),
    ]:
        strategy = get_timestep_imputation_strategy(data, column)
        assert isinstance(strategy, strategy_class)


def test_get_timestep_imputation_strategy_error():
    with pytest.raises(ValueError):
        get_timestep_imputation_strategy(date_data, "nonexistent_column")


@pytest.mark.parametrize(
    "data,column",
    [
        (pd.DataFrame({"non_datetime": ["a", "b", "c"]}), "non_datetime"),
        (pd.DataFrame({"non_integer": ["1.1", "2.2", "3.3"]}), "non_integer"),
    ],
)
def test_check_timeseries_type_error(data, column):
    with pytest.raises(ValueError):
        check_timeseries_type(data, column)
