import os
import pickle
import logging
import pandas as pd
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    logger = logging.getLogger(__name__)

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    with open(model_dir, 'rb') as f_in:
        model = pickle.load(f_in)

    data["predict"] = model.predict_proba(data)[:, 1]

    os.makedirs(output_dir, exist_ok=True)
    predict_path = os.path.join(output_dir, "predictions.csv")
    data.to_csv(predict_path, index=False)
    logger.info(f"predict completed and saved in {predict_path}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    predict()
