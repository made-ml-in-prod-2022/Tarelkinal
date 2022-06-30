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
    MODEL_DIR,
    NETWORK,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
)


with DAG(
        "train_model",
        default_args=DEFAULT_ARGS,
        schedule_interval="0 0 * * 7",
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

    check_target_sensor = FileSensor(
        task_id="wait-for-target",
        filepath=str("opt/airflow" / Path(RAW_DATA_DIR) / "target.csv"),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    dummy_wait_data = DummyOperator(
        task_id='dummy-wait-data',
        default_args=DEFAULT_ARGS,
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

    split_data = DockerOperator(
        image="airflow-split-data",
        command=f"--input-dir {PROCESSED_DATA_DIR}",
        network_mode=NETWORK,
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )

    train_model = DockerOperator(
        image="airflow-train-model",
        command=f"--input-dir {PROCESSED_DATA_DIR} --output-dir {MODEL_DIR}",
        network_mode=NETWORK,
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )

    validation = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {PROCESSED_DATA_DIR} --model-dir {MODEL_DIR}",
        network_mode=NETWORK,
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )

    [check_data_sensor, check_target_sensor] >> dummy_wait_data
    dummy_wait_data >> preprocess >> split_data >> train_model >> validation
