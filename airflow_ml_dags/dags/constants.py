import datetime

from airflow.models import Variable
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}
START_DATE = days_ago(8)
DATA_VOLUME_DIR = Variable.get("data_path")
MODEL_PATH = Variable.get("model_path")
NETWORK = Variable.get("network")
RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"
PREDICTIONS_DIR = "/data/predictions/{{ ds }}"
