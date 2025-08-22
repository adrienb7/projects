from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests

default_args = {
    "owner": "mlops_team",
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="accidents_training_pipeline_api",
    default_args=default_args,
    description="Pipeline Airflow pour déclencher l'entraînement du modèle accidents via API",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["accidents", "training", "api"],
) as dag:

    @task
    def choose_dataset():
        """
        Sélectionne un dataset parmi ceux disponibles.
        Ici les datasets sont enregistrés au format csv.
        """
        import random
        datasets = [
            "/opt/data/accidents_dataset_v1.csv",
            "/opt/data/accidents_dataset_v2.csv",
        ]
        selected_dataset = random.choice(datasets)
        return selected_dataset

    @task
    def trigger_training(dataset_path: str):
        """
        Déclenche l'entraînement du modèle en appelant l'API FastAPI prévue à cet effet.
        """
        api_url = "http://0.0.0.0:8089/api/v1/train"
        payload = {"dataset_path": dataset_path}

        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            print("Entraînement lancé avec succès.")
        else:
            raise Exception(f"Erreur lors de l'appel à l'API : {response.status_code} - {response.text}")

    # Exécution du pipeline
    dataset_path = choose_dataset()
    trigger_training(dataset_path)
