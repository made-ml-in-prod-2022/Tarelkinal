import os
import logging
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
import click


@click.command("train_model")
@click.option("--input-dir")
@click.option("--output-dir")
def train_model(input_dir: str, output_dir):
    logger = logging.getLogger(__name__)

    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    model = LogisticRegression()
    model.fit(data, target)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")

    with open(model_path, 'wb') as f_out:
        pickle.dump(model, f_out)

    logger.info(f'model trained and saved in {model_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_model()
