import os
from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

from constants import (
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MODEL_PATH,
    NETWORK,
    RAW_DATA_DIR,
    PREDICTIONS_DIR,
    START_DATE,
    PROCESSED_DATA_DIR
)


with DAG(
        "predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:

    check_data_sensor = FileSensor(
        task_id="wait-for-data",
        filepath=str(Path(RAW_DATA_DIR) / "data.csv"),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    check_model_sensor = FileSensor(
        task_id="wait-for-model",
        filepath=MODEL_PATH,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {RAW_DATA_DIR} --output-dir {PROCESSED_DATA_DIR}",
        network_mode=NETWORK,
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {PROCESSED_DATA_DIR} --output-dir {PREDICTIONS_DIR} --model-dir {MODEL_PATH}",
        network_mode=NETWORK,
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )

    [check_data_sensor, check_model_sensor] >> preprocess >> predict
