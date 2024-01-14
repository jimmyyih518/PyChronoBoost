# PyChronoBoost

Automated Time Series Feature Engineering

PyChronoBoost is a sophisticated Python package designed for time series data analysis. It offers robust functionality for imputing missing values, generating and selecting features, and handling various complexities associated with time series data.

## Features

- **Time Series Imputation**: Efficiently handles missing data in time series, both in terms of values and time steps.
- **Feature Generation**: Automatic generation of relevant features from time series data.
- **Feature Selection**: Utilizes advanced algorithms like XGBoost to select the most significant features for your analysis.

## Installation

Install PyChronoBoost using pip:

```bash
pip install git+https://github.com/jimmyyih518/PyChronoBoost
```

## Quick Start

Here's a quick example to get you started with PyChronoBoost:
```
import pandas as pd
from pychronoboost.timeseries import TimeSeriesData

# Sample time series data
data = pd.DataFrame({
    'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
    'value': [1, None, 3, 4, 5]
})

# Initialize TimeSeriesData object
ts_data = TimeSeriesData(data, timestep_column='time')

# Process features
processed_data = ts_data.process_timeseries_features(
    feature_columns=['value'],
    target_column='value',
    value_impute_strategy='last',
    max_window_size=3,
    feature_selector_model='XGB',
    max_features=5
)

print(processed_data)

```

## Documentation
For more detailed information and examples, please refer to the [example notebook](https://github.com/jimmyyih518/PyChronoBoost/blob/main/doc/example.ipynb).

## Contributing
Contributions to PyChronoBoost are welcome! Please contact the author for more information.

## License
PyChronoBoost is released under the Apache License.

## Contact
For any queries or suggestions, feel free to open an issue on GitHub or contact me directly at [jimmyyih@ualberta.com].

*I hope PyChronoBoost makes your time series data analysis more efficient and insightful!*