import pandas as pd

TIMESTEP_INTEGER = "integer"
TIMESTEP_FLOAT = "float"
TIMESTEP_DATE = "date"
TIMESTEP_DATETIME = "datetime"


def check_timeseries_type(data: pd.DataFrame, column: str) -> str:
    """
    Check the type of a timeseries column in a DataFrame.

    :param data: The DataFrame containing the timeseries data.
    :param column: The name of the column to check.
    :return: The type of the timeseries column ('date', 'datetime', 'integer', or 'float').
    :raises ValueError: If the column type is not supported or cannot be converted.
    """
    if column not in data.columns:
        raise ValueError(f"The column '{column}' is not in the DataFrame.")

    if pd.api.types.is_integer_dtype(data[column]):
        return TIMESTEP_INTEGER
    elif pd.api.types.is_float_dtype(data[column]):
        return TIMESTEP_FLOAT
    else:
        try:
            converted = pd.to_datetime(data[column])
            # Check if all times are midnight, indicating a 'date' type
            if all(time == pd.Timestamp(0).time() for time in converted.dt.time):
                data[column] = converted
                return TIMESTEP_DATE
            else:
                data[column] = converted
                return TIMESTEP_DATETIME
        except ValueError:
            raise ValueError(
                "The column must contain integers, floats, or be convertible to dates/datetime."
            )


# Example usage in TimeSeriesData class:
# ts_type = check_timeseries_type(self.data, self.timestep_column)