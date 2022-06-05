import pytest
import pandas as pd
import numpy as np
import logging

from sklearn.datasets import make_classification


@pytest.fixture()
def logger():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    return logger


@pytest.fixture()
def dataset():
    df_size = 1000
    df = pd.DataFrame({
        'id': [i for i in range(df_size)],
        'cat_feature': [0]*500 + [1]*498 + [2]*2,
        'target': np.random.choice([0, 1], size=df_size),
        'det_target': [0]*400 + [1]*600,
    })
    return df


@pytest.fixture()
def real_dataset():
    X, y = make_classification(random_state=0)
    feature_name_list = [f'f_{i}' for i in range(X.shape[1])]
    target_name = ['target']
    df = pd.DataFrame(
        np.concatenate([X, y.reshape(-1, 1)], axis=1),
        columns=feature_name_list + target_name
    )

    return df
