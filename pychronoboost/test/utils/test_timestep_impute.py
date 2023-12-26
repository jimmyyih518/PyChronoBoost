import pytest
import pandas as pd
import numpy as np
from pychronoboost.src.utils.timestep_impute import (
    DateImputation,
    DateTimeImputation,
    IntegerImputation,
    FloatImputation,
    get_timestep_imputation_strategy,
)

# Sample data for tests
date_data = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"])})
datetime_data = pd.DataFrame(
    {"datetime": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:02"])}
)
integer_data = pd.DataFrame({"integer": [1, 3]})
float_data = pd.DataFrame({"float": [1.0, 1.2]})


def test_date_imputation():
    strategy = DateImputation()
    result = strategy.impute(date_data, "date")
    assert len(result) == 3
    assert result["date"].isna().sum() == 0


def test_datetime_imputation():
    strategy = DateTimeImputation()
    result = strategy.impute(datetime_data, "datetime")
    assert len(result) == 3
    assert result["datetime"].isna().sum() == 0


def test_integer_imputation():
    strategy = IntegerImputation()
    result = strategy.impute(integer_data, "integer")
    assert len(result) == 3
    assert result["integer"].isna().sum() == 0


def test_float_imputation():
    # Adjusting the input data
    adjusted_float_data = pd.DataFrame({"float": [1.0, 1.3]})

    strategy = FloatImputation()
    result = strategy.impute(adjusted_float_data, "float")
    assert len(result) > 2


def test_get_timestep_imputation_strategy():
    date_strategy = get_timestep_imputation_strategy(date_data, "date")
    assert isinstance(date_strategy, DateImputation)

    datetime_strategy = get_timestep_imputation_strategy(datetime_data, "datetime")
    assert isinstance(datetime_strategy, DateTimeImputation)

    integer_strategy = get_timestep_imputation_strategy(integer_data, "integer")
    assert isinstance(integer_strategy, IntegerImputation)

    float_strategy = get_timestep_imputation_strategy(float_data, "float")
    assert isinstance(float_strategy, FloatImputation)

    with pytest.raises(ValueError):
        get_timestep_imputation_strategy(date_data, "unknown")
