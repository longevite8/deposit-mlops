import numpy as np
import pandas as pd
from config import (
    RANDOM_STATE,
    DATE_COLUMN,
    TARGET_COLUMN,
    START_DATE,
    N_DAYS,
    GAMMA_SHAPE,
    GAMMA_SCALE,
)


def extract_data() -> pd.DataFrame:
    """Generate synthetic data for daily deposits."""

    np.random.seed(RANDOM_STATE)

    dates = pd.date_range(
        start=START_DATE,
        periods=N_DAYS,
    )

    df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            TARGET_COLUMN: np.random.gamma(
                shape=GAMMA_SHAPE,
                scale=GAMMA_SCALE,
                size=len(dates),
            ),
        }
    )

    return df
