from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pychronoboost.pychronoboost.utils.helpers import (
    check_timeseries_type,
    TIMESTEP_INTEGER,
    TIMESTEP_FLOAT,
    TIMESTEP_DATE,
    TIMESTEP_DATETIME,
)


class TimeStepImputationStrategy(ABC):
    @abstractmethod
    def impute(self, data: pd.DataFrame, timestep_column: str):
        """
        Impute missing timesteps in the data.
        :param data: Pandas DataFrame with missing timesteps.
        :param timestep_column: The column name of the timestep.
        """
        pass


class DateImputation(TimeStepImputationStrategy):
    def impute(self, data: pd.DataFrame, timestep_column: str):
        data = data.set_index(timestep_column)
        data = data.asfreq("D")
        data.reset_index(inplace=True)
        return data


class DateTimeImputation(TimeStepImputationStrategy):
    def impute(self, data: pd.DataFrame, timestep_column: str):
        data = data.set_index(timestep_column)
        data = data.asfreq("S")
        data.reset_index(inplace=True)
        return data


class IntegerImputation(TimeStepImputationStrategy):
    def impute(self, data: pd.DataFrame, timestep_column: str):
        data = data.set_index(timestep_column)
        data = data.reindex(np.arange(data.index.min(), data.index.max() + 1))
        data.reset_index(inplace=True)
        return data


class FloatImputation(TimeStepImputationStrategy):
    def impute(self, data: pd.DataFrame, timestep_column: str):
        data = data.set_index(timestep_column)
        min_value, max_value = data.index.min(), data.index.max()
        new_index = np.arange(min_value, max_value, 0.1)
        data = data.reindex(new_index)
        data.reset_index(inplace=True)
        return data


def get_timestep_imputation_strategy(data: pd.DataFrame, timestep_column: str):
    timestep_type = check_timeseries_type(data, timestep_column)
    if timestep_type == TIMESTEP_DATE:
        return DateImputation()
    elif timestep_type == TIMESTEP_DATETIME:
        return DateTimeImputation()
    elif timestep_type == TIMESTEP_INTEGER:
        return IntegerImputation()
    elif timestep_type == TIMESTEP_FLOAT:
        return FloatImputation()
    else:
        raise ValueError(f"Unsupported timestep type: {timestep_type}")


# Usage Example:
# strategy = get_timestep_imputation_strategy(data, 'timestamp')
# imputed_data = strategy.impute(data, 'timestamp')
