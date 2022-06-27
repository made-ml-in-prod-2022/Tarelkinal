import os
import logging
import pickle

import pandas as pd
from sklearn.metrics import roc_auc_score
import click


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir):
    logger = logging.getLogger(__name__)

    data = pd.read_csv(os.path.join(input_dir, "test_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "test_target.csv"))

    logger.info(f'loaded test data with shape {data.shape}')
    logger.info(f'loaded test target with shape {target.shape}')

    with open(os.path.join(model_dir, "model.pkl"), 'rb') as f_in:
        model = pickle.load(f_in)

    score = roc_auc_score(target, model.predict_proba(data)[:, 1])
    logger.info(f'roc_auc_score {score}')

    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "score.txt"), 'wb') as f_out:
        f_out.write(str(score).encode())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    validate()
