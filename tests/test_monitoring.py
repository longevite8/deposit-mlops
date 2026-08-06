import numpy as np
import pytest

from business.monitoring import prepare_prediction_histogram_values


def test_prediction_histogram_accepts_numpy_array():
    predictions = np.array([10.0, 20.0, 30.0])

    values = prepare_prediction_histogram_values(predictions)

    assert isinstance(values, np.ndarray)
    assert values.tolist() == [10.0, 20.0, 30.0]


def test_prediction_histogram_accepts_pandas_series():
    import pandas as pd

    predictions = pd.Series([10.0, 20.0, 30.0])

    values = prepare_prediction_histogram_values(predictions)

    assert isinstance(values, np.ndarray)
    assert values.tolist() == [10.0, 20.0, 30.0]


def test_prediction_histogram_rejects_multidimensional_array():
    predictions = np.array([[10.0, 20.0], [30.0, 40.0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        prepare_prediction_histogram_values(predictions)
