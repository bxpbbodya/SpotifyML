from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from datetime import timedelta
import json
import os

PROJECT_PATH = "/opt/airflow/project"
METRICS_PATH = os.path.join(PROJECT_PATH, "metrics.json")

default_args = {
    "owner": "student",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

def check_metric(**kwargs):
    if not os.path.exists(METRICS_PATH):
        return "stop_pipeline"

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    r2 = metrics.get("r2_test", 0)

    if r2 >= 0.5:
        return "register_model"

    return "stop_pipeline"


with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="@daily",   # Continuous Training
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "ct", "level2"],
) as dag:

    # 1️⃣ Sensor — перевірка наявності підготовлених даних
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath="/opt/airflow/project/data/prepared/train.csv",
        poke_interval=30,
        timeout=600,
        mode="poke",
    )

    # 2️⃣ Data preparation (DVC)
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow/project && dvc repro prepare",
    )

    # 3️⃣ Model training
    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/project && python src/train.py data/prepared",
    )

    # 4️⃣ Evaluation (логічний етап)
    evaluate = EmptyOperator(task_id="evaluate_model")

    # 5️⃣ Branching (Quality Gate)
    branch = BranchPythonOperator(
        task_id="check_performance",
        python_callable=check_metric,
    )

    # 6️⃣ Registration (MLflow Registry відбувається в train.py)
    register = EmptyOperator(task_id="register_model")

    stop = EmptyOperator(task_id="stop_pipeline")

    # DAG structure
    wait_for_data >> prepare_data >> train_model >> evaluate >> branch
    branch >> register
    branch >> stop