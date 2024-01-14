import pytest
import pandas as pd
import numpy as np
from pychronoboost.impute.value_impute import (
    get_value_imputation_strategy,
    LastValueImputation,
    ZeroImputation,
    LinearImputation,
)


# Test for correct strategy creation
def test_correct_strategy_creation():
    assert isinstance(get_value_imputation_strategy("last"), LastValueImputation)
    assert isinstance(get_value_imputation_strategy("zero"), ZeroImputation)
    assert isinstance(get_value_imputation_strategy("linear"), LinearImputation)


# Test for exception on unrecognized strategy
def test_unrecognized_strategy():
    with pytest.raises(NotImplementedError):
        get_value_imputation_strategy("unknown")


def test_last_value_imputation():
    strategy = get_value_imputation_strategy("last")
    data = pd.DataFrame({"col1": [1, 2, None, 4]})
    data["imputed"] = strategy.impute(data["col1"])
    assert data["imputed"].isnull().sum().sum() == 0


def test_zero_imputation():
    strategy = get_value_imputation_strategy("zero")
    data = pd.DataFrame({"col1": [1, None, 3, 4]})
    data["imputed"] = strategy.impute(data["col1"])
    assert (data["imputed"] == [1, 0, 3, 4]).all()


def test_linear_imputation():
    strategy = get_value_imputation_strategy("linear")
    data = pd.DataFrame({"col1": [1, None, 3, 4]})
    data["imputed"] = strategy.impute(data["col1"])
    assert data["imputed"][1] == 2  # Assuming linear interpolation between 1 and 3
