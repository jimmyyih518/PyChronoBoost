def test_time_series_data_importable():
    try:
        from pychronoboost.data import TimeSeriesData
    except ImportError as e:
        assert False, f"Failed to import TimeSeriesData, {e}"
