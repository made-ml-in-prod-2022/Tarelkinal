import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command("train_model")
@click.option("--input-dir")
def split(input_dir: str):
    logger = logging.getLogger(__name__)

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.25, random_state=0)

    logger.info(f'X_train shape is {X_train.shape}')
    logger.info(f'y_train shape is {y_train.shape}')
    logger.info(f'X_test shape is {X_test.shape}')
    logger.info(f'y_test shape is {y_test.shape}')

    X_train.to_csv(os.path.join(input_dir, "train_data.csv"), index=False)
    X_test.to_csv(os.path.join(input_dir, "test_data.csv"), index=False)
    y_train.to_csv(os.path.join(input_dir, "train_target.csv"), index=False)
    y_test.to_csv(os.path.join(input_dir, "test_target.csv"), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    split()
