import os
from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

from constants import (
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MODEL_PATH,
    NETWORK,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    START_DATE,
)


with DAG(
        "predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:

    check_data_sensor = FileSensor(
        task_id="wait-for-data",
        filepath=str(Path("opt/airflow") / Path(PROCESSED_DATA_DIR) / "data.csv"),
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

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {PROCESSED_DATA_DIR} --output-dir {PREDICTIONS_DIR} --model-dir {MODEL_PATH}",
        network_mode=NETWORK,
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )

    dummy_start = DummyOperator(
        task_id='dummy-predict-start',
        default_args=DEFAULT_ARGS,
    )

    [check_data_sensor, check_model_sensor] >> dummy_start >> predict
