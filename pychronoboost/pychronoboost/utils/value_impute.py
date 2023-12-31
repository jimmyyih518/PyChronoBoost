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
        return data.fillna(method="ffill")


class ZeroImputation(ValueImputationStrategy):
    def impute(self, data):
        return data.fillna(0)


class LinearImputation(ValueImputationStrategy):
    def impute(self, data):
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError(f'Linear interpolation does not apply to non-numeric column')
        return (
            data.interpolate(method="linear")
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
