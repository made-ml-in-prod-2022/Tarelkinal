import os
import logging

import click
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    logger = logging.getLogger(__name__)

    X, y = make_classification(random_state=0)
    feature_name_list = [f'f_{i}' for i in range(X.shape[1])]
    target_name = ['target']
    df = pd.DataFrame(
        np.concatenate([X, y.reshape(-1, 1)], axis=1),
        columns=feature_name_list + target_name
    )

    os.makedirs(output_dir, exist_ok=True)
    df[feature_name_list].to_csv(os.path.join(output_dir, "data.csv"), index=False)
    df[target_name].to_csv(os.path.join(output_dir, "target.csv"), index=False)

    logger.info(f'data save in {os.path.join(output_dir, "data.csv")}')
    logger.info(f'data save in {os.path.join(output_dir, "target.csv")}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    download()
