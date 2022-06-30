import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import click
import logging


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir):
    logger = logging.getLogger(__name__)

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    logger.info(f'data loaded with shape {data.shape}')
    logger.info(f'target loaded with shape {target.shape}')

    poly = PolynomialFeatures()
    data_prep = pd.DataFrame(poly.fit_transform(data))

    logger.info(f'data_prep shape {data.shape}')

    os.makedirs(output_dir, exist_ok=True)
    data_prep.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    preprocess()
