from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class ValueImputationStrategy(ABC):
    @abstractmethod
    def impute(self, data):
        """
        Impute missing values in the data.
        :param data: Pandas DataFrame with missing values.
        """
        pass


class LastValueImputation(ValueImputationStrategy):
    def impute(self, data):
        return data.fillna(method="ffill").fillna(method="bfill")


class ZeroImputation(ValueImputationStrategy):
    def impute(self, data):
        return data.fillna(0)


class LinearImputation(ValueImputationStrategy):
    def impute(self, data):
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError(
                f"Linear interpolation does not apply to non-numeric column"
            )
        return (
            data.interpolate(method="linear")
            .fillna(method="ffill")
            .fillna(method="bfill")
        )


def get_value_imputation_strategy(strategy: str) -> ValueImputationStrategy:
    value_imputation_strategy = {
        "last": LastValueImputation(),
        "zero": ZeroImputation(),
        "linear": LinearImputation(),
    }

    if strategy not in value_imputation_strategy:
        raise NotImplementedError(
            f"Strategy {strategy} not available"
        )

    return value_imputation_strategy[strategy]

# Usage
# imputer = get_value_imputation_strategy("linear")
# Now you can use `imputer.impute(data)`