from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from docker.types import Mount

from constants import (
    DATA_VOLUME_DIR, DEFAULT_ARGS, RAW_DATA_DIR, START_DATE
)


with DAG(
        "download_data",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command=f"{RAW_DATA_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=str(Path(DATA_VOLUME_DIR)), target="/data", type='bind')]
    )
    dummy_start = DummyOperator(
        task_id='dummy-download-start',
        default_args=DEFAULT_ARGS,
    )

    dummy_start >> download
