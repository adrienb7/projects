from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from minio import Minio
from io import BytesIO
import pickle
#from mlflow import MlflowClient, sklearn
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import  os, requests

import time
import random
import logging

# MINIO
MINIO_ROOT_USER = "minio"
MINIO_ROOT_PASSWORD = "minio123"
MINIO_ACCESS_KEY = "XeAMQQjZY2pTcXWfxh4H" 
MINIO_SECRET_ACCESS_KEY = "wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ"
#NINIO_REGION=us-east-1
MINIO_RAW_BUCKET_NAME_RAW = "accidents-raw-csv"
MINIO_RAW_BUCKET_NAME_PREPROCESSED = "accidents-preprocessed-csv"
MINIO_API_ADDRESS = 'minio:9000' 
MINIO_STORAGE_USE_HTTPS = False 
MINIO_CONSOLE_ADDRESS = '0.0.0.0:9001' 
MINIO_API_PORT = 9000 
MINIO_CONSOLE_PORT = 9001

# BUCKETS
RAW_BUCKET_NAME = "accidents-raw-csv"
PREPROCESSED_BUCKET_NAME = "accidents-preprocessed-csv"
TRAIN_BUCKET_NAME = "accidents-train-csv"
MODEL_BUCKET_NAME = "accidents-models"

# FILESYSTEM CSV FILES
LOCAL_DIR = '/app/data/raw/'
LOCAL_DIR_PREPROCESSED = '/app/data/preprocessed/'

# APIKEY
API_KEY = Variable.get('api-key')

# Jeu de données
#from sklearn.datasets import make_classification
#X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

default_args = {
        'owner': 'airflow',
        'start_date': datetime(2025, 5, 15, 0, 0),  # date fixe dans le passé
    }

models = {
    "RandomForest": {
        "pipeline": Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        "param_grid": {"clf__n_estimators": [10], "clf__max_depth": [None]}
    },
    #"SVC": {
    #    "pipeline": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
    #    "param_grid": {"clf__C": [1], "clf__kernel": ["rbf"]}
    #},
    "MLP": {
        "pipeline": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=300))]),
        "param_grid": {"clf__hidden_layer_sizes": [(10,)], "clf__alpha": [0.001]}
    }
}

# Put csv in Minio
def load_csv_in_minio():
    try:
        # laod from file sytrem to minio s3
        RAW_BUCKET_NAME = "accidents-raw-csv"
        minio_client = Minio('minio:9000', access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY, secure=False) 

        # Vérifier si le bucket existe, sinon le créer
        if not minio_client.bucket_exists(RAW_BUCKET_NAME):
            minio_client.make_bucket(RAW_BUCKET_NAME)
            print(f"Bucket '{RAW_BUCKET_NAME}' créé.")

        # Lister tous les fichiers CSV dans le répertoire local
        for filename in os.listdir(LOCAL_DIR):
            if filename.endswith('.csv'):
                local_file_path = os.path.join(LOCAL_DIR, filename)
                # Write chaque fichier dans MinIO
                print(f"Uploading {filename} to MinIO...")
                minio_client.fput_object(RAW_BUCKET_NAME, filename, local_file_path)
                print(f"File {filename} uploaded successfully to MinIO.")
                # delete from local path
                #os.remove(local_file_path)
       
    except Exception as e:
        raise e
    

# Fonction pour récupérer les fichiers CSV et les grouper par année
def merge_files_by_year():
    minio_client = Minio('minio:9000', access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY, secure=False)

    # Dictionnaire pour stocker les fichiers par année
    files_by_year = {}

    # Liste les fichiers dans le bucket MinIO
    for filename in minio_client.list_objects(RAW_BUCKET_NAME):
        if filename.object_name.endswith('.csv'):
            year = filename.object_name.split('-')[-1].split('.')[0]  # Extraire l'année du nom du fichier
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(filename.object_name)

    all_merged_dfs = []  # Liste de tous les DataFrames fusionnés par année

     # Pour chaque groupe d'années, récupérer les fichiers, les fusionner et les supprimer
    for year, files in files_by_year.items():
        print(f"Merging files for year {year}...")

        # Dictionnaire pour stocker les DataFrames
        dfs = {'lieux': None, 'vehicules': None, 'usagers': None, 'caracteristiques': None}

        # Télécharger les fichiers et les charger dans des DataFrames
        for file in files:
            print(f"=> Downloading {file} from MinIO...")
            #minio_client.fput_object(RAW_BUCKET_NAME, file, f"{LOCAL_DIR}{file}")  # Récupérer le fichier en mémoire
            obj = minio_client.get_object(RAW_BUCKET_NAME, file)
            #
            df = pd.read_csv(obj, dtype=str, sep=";")
            #df = pd.read_csv(BytesIO(data.read()), on_bad_lines='warn')  # Charger en DataFrame
            #df = pd.read_csv(f"{LOCAL_DIR}{file}", dtype=str, sep=";")
            print(f"=> year =:{year}")
            if 'lieux' in file:
                dfs['lieux'] = df
                print(f"=> lieux :{df.head()} ")
            elif 'vehicules' in file:
                print(f"=> vehicules :{df.head()} ")
                dfs['vehicules'] = df
            elif 'usagers' in file:
                print(f"=> usagers :{df.head()} ")
                dfs['usagers'] = df
            elif 'caracteristiques' in file:
                print(f"=> caracteristiques :{df.head()} ")
                if '2022' in year:
                    dfs['caracteristiques'] = df.rename(columns={"Accident_Id": "Num_Acc"}, inplace=True)
                dfs['caracteristiques'] = df
        # Fusion
        merged_df = dfs['caracteristiques'].merge(dfs['lieux'], on='Num_Acc').merge(dfs['vehicules'], on='Num_Acc').merge(dfs['usagers'], on=['Num_Acc', 'id_vehicule'])

        # save to minio S3
        print(f"=> push in Minio df_accidents_{year}.csv")
        csv_bytes = merged_df.to_csv(sep=";").encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)
        minio_client.put_object(PREPROCESSED_BUCKET_NAME, f"df_accidents_{year}.csv", data=csv_buffer, length=len(csv_bytes), content_type='application/csv')

        # Ajouter une colonne année si utile
        merged_df['year'] = year
        # Ajouter à la liste
        all_merged_dfs.append(merged_df)

    # Concaténer tous les DataFrames en un seul
    df_accidents = pd.concat(all_merged_dfs, ignore_index=True)

    # Sauvegarde locale puis upload dans MinIO (optionnel)
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'df_accidents.csv')
    df_accidents.to_csv(output_file, sep=";", index=False)
    minio_client.fput_object(PREPROCESSED_BUCKET_NAME, 'df_accidents.csv', output_file)
    print("Fichier global df_accidentf.csvgénéré et envoyé dans MinIO.")


def clean_data():
    #
    minio_client = Minio('minio:9000', access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY, secure=False)
    # Dowload df_accidents.csv in Minio
    obj = minio_client.get_object(PREPROCESSED_BUCKET_NAME, 'df_accidents.csv')
    #
    data = pd.read_csv(obj, dtype=str, sep=";")

    # 1. Remplacer les virgules par des points dans les colonnes décimales
    data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    
    # 2. Convertir toutes les colonnes numériques au bon format
    # Nous utilisons `errors='ignore'` pour que les colonnes non numériques ne soient pas affectées
    data = data.apply(pd.to_numeric, errors='ignore')
    
    # 3. Supprimer les colonnes spécifiques
    columns_to_drop = ['Num_Acc', 'v1', 'v2', 'Acc', 'dep', 'com', 'int', 'adr', 'voie', 'lartpc', 
                       'larrout', 'id_vehicule', 'trajet', 'pr', 'pr1', 'num_veh_x', 'occutc', 'num_veh_y', 
                       'actp', 'etatp', 'id_usager', 'year']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # 4. Décrémenter les valeurs de la colonne 'grav' de 1
    if 'grav' in data.columns:
        data['grav'] = data['grav'] - 1
    
    # 5. Nettoyage de la colonne 'nbv' : supprimer les valeurs inférieures à 0 et les valeurs non numériques
    if 'nbv' in data.columns:
        # Convertir 'nbv' en numérique (les erreurs seront transformées en NaN)
        data['nbv'] = pd.to_numeric(data['nbv'], errors='coerce')
        
        # Supprimer les lignes où 'nbv' est inférieure à 0 ou NaN
        data = data[data['nbv'] >= 0]
        data = data.dropna(subset=['nbv'])
        
        # Convertir 'nbv' en entier (si nécessaire)
        data['nbv'] = data['nbv'].astype(int)

        # Convertir les hh:mm en mm
        data['hrmn'] = data['hrmn'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # Afficher les premières lignes pour vérifier le résultat
    print(data.head())
    #
    data = data.dropna()
    # Feature && target
    X = data.drop(['grav'], axis=1).astype('float')
    y = data['grav']

    # Division de notre dataset en deux. Pour la cross validation (train_test) 90 % des données et 10% des données pour tester les modèles finaux (global_test)
    X_train_test, X_global_test, y_train_test, y_global_test = train_test_split(X, y, test_size=0.10, random_state=42)

    # Sauvegarder clean data
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'df_accident_clean.csv')
    data.to_csv(output_file, sep=";", index=False)
    minio_client.fput_object(PREPROCESSED_BUCKET_NAME, 'df_accident_clean.csv', output_file)

    # Sauvegarder les splits train's
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'X_train_test.csv')
    X_train_test.to_csv(output_file, sep=";", index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'X_train_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}X_train_test.csv")
    #
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'X_global_test.csv')
    X_global_test.to_csv(output_file, sep=";", index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'X_global_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}X_global_test.csv")
    #
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'y_train_test.csv')
    y_train_test.to_csv(output_file, sep=";", index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'y_train_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}y_train_test.csv")
    #
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'y_global_test.csv')
    y_global_test.to_csv(output_file, sep=";", index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'y_global_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}y_global_test.csv")

     

def search_best_params(**kwargs):
    #
    minio_client = Minio('minio:9000', access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY, secure=False)
    # S3
    obj = minio_client.get_object(TRAIN_BUCKET_NAME, 'X_train_test.csv')
    X_train = pd.read_csv(obj, dtype=str, sep=";")
    obj = minio_client.get_object(TRAIN_BUCKET_NAME, 'y_train_test.csv')
    y_train = pd.read_csv(obj, dtype=str, sep=";").values.ravel()
    #
    model_name = kwargs["model_name"]
    config = models[model_name]
    grid = GridSearchCV(config["pipeline"], config["param_grid"], scoring="f1_weighted", cv=2)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    kwargs["ti"].xcom_push(key="best_params", value=best_params)
    

def train_best_model(**kwargs):
    #
    minio_client = Minio('minio:9000', access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY, secure=False)
    # S3
    obj = minio_client.get_object(TRAIN_BUCKET_NAME, 'X_train_test.csv')
    X_train = pd.read_csv(obj, dtype=str, sep=";")
    obj = minio_client.get_object(TRAIN_BUCKET_NAME, 'y_train_test.csv')
    y_train = pd.read_csv(obj, dtype=str, sep=";").values.ravel()
    #
    model_name = kwargs["model_name"]
    best_params = kwargs["ti"].xcom_pull(key="best_params", task_ids=f"search_params_{model_name}")
    pipeline = models[model_name]["pipeline"].set_params(**best_params)
    pipeline.fit(X_train, y_train)
    
    # Write mosel into S3
    bytes_file = pickle.dumps(pipeline)
    buffer = BytesIO(bytes_file)
    minio_client.put_object(
        bucket_name=MODEL_BUCKET_NAME, 
        object_name=f"{model_name}_model.pkl", 
        data=buffer,
        length=len(bytes_file),
        content_type="application/octet-stream"
        )
    

def log_model_mlflow(**kwargs):
    # getsion des traces
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # model name
    model_name = kwargs["model_name"]

    # Load model on S3
    minio_client = Minio(
        'minio:9000',
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_ACCESS_KEY,
        secure=False)
    # S3
    try:
        # Load model
        logger.info(f"📦 Chargement du modèle {model_name} depuis MinIO...")
        model_obj = minio_client.get_object(MODEL_BUCKET_NAME, f"{model_name}_model.pkl")
        model = pickle.loads(model_obj.read())

        # Load data
        X_obj = minio_client.get_object(TRAIN_BUCKET_NAME, "X_global_test.csv")
        X_train = pd.read_csv(X_obj, dtype=str, sep=";")

        y_obj = minio_client.get_object(TRAIN_BUCKET_NAME, "y_global_test.csv")
        y_train = pd.read_csv(y_obj, dtype=str, sep=";").values.ravel()
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle ou des données : {e}")
        raise

    # MLflow tracking
    #mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = MlflowClient()

    try:
        mlflow.set_experiment("Gravite_Accidents")
    except Exception as e:
        logger.warning(f"⚠️ Problème lors de la définition de l'expérience : {e}")

    try:
        with mlflow.start_run(run_name=model_name):
            logger.info("🚀 Début de l'entraînement et du log MLflow")
            # Cross-validation avec plusieurs métriques
            scoring = ["accuracy", "f1_weighted", "recall_weighted", "precision_weighted"]
            scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=2)
            # Log de toutes les métriques moyennes
            for metric in scoring:
                mean_score = scores[f"test_{metric}"].mean()
                log_with_retry(f"cv_{metric}", mean_score, client)
            
            #scores = cross_validate(model, X_train, y_train, scoring=["accuracy", "f1_weighted"], cv=3)

            #log_with_retry("cv_accuracy", scores["test_accuracy"].mean(), client)
            #log_with_retry("cv_f1_weighted", scores["test_f1_weighted"].mean(), client)

            try:
                logger.info("🧠 Log des paramètres du modèle")
                mlflow.log_params(model.get_params())
            except Exception as e:
                logger.warning(f"⚠️ Impossible de log les paramètres : {e}")

            logger.info("📤 Log du modèle dans MLflow...")
            safe_log_model(model, "model", registered_model_name=model_name, retries=4)
            #mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            logger.info("✅ Modèle loggé avec succès.")

    except Exception as e:
        logger.error(f"❌ Échec de la tâche MLflow : {e}")
        raise
    

def log_with_retry(key, value, client, retries=5, wait=15):
    for attempt in range(retries):
        try:
            mlflow.log_metric(key, value)
            logging.info(f"📈 Metric '{key}' enregistrée : {value}")
            return
        except Exception as e:
            logging.warning(f"[{attempt+1}/{retries}] ⚠️ Erreur lors du log de '{key}' : {e}")
            time.sleep(wait + random.random())
    raise RuntimeError(f"❌ Impossible de log la métrique '{key}' après {retries} tentatives.")


def safe_log_model(model, name, registered_model_name=None, retries=3):
    for i in range(retries):
        try:
            mlflow.sklearn.log_model(model, name, registered_model_name=registered_model_name)
            return
        except Exception as e:
            logging.warning(f"[Retry {i+1}] Erreur lors du log_model : {e}")
            time.sleep(3 + i * 2)
    raise RuntimeError("❌ Impossible de log le modèle après plusieurs tentatives.")


def reload_model_api(**kwargs):
    model_name = kwargs["model_name"]
    #
    url = f"http://app/api/v1/reload_model/{model_name}"
    #
    headers = {
        "X-Api-Key": API_KEY
    }
    #
    try:
        response = requests.post(url, headers=headers)
        #
        print(response.status_code)
        print(response.json())
    except Exception as e:
        raise RuntimeError(f"Échec reload {model_name} FastAPI : {e}")
    
    
'''
Sélectionne dynamiquement le meilleur modèle MLflow selon des métriques pondérées

Ignore tous les runs sans modèle loggé ou avec métriques incomplètes

Enregistre le modèle comme BestModel dans le Model Registry

Le promeut en "Production" automatiquement
'''   
    
def select_best_model():
    #
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Gravite_Accidents")
    if not experiment:
        raise ValueError("L'expérience 'Gravite_Accidents' est introuvable.")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    best_run = None
    best_score = -1

    # Pondération des métriques — à ajuster selon ton objectif
    metric_weights = {
        "cv_accuracy": 0.25,
        "cv_f1_weighted": 0.35,
        "cv_recall_weighted": 0.2,
        "cv_precision_weighted": 0.2
    }

    # Exemple : entraîner et conserver tous les modèles testés 
    evaluated_models = {}
    runs_with_models = []

    for run in runs:
        # on reload les models pour pouvoir associer le model.pkl au BestModel
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        #
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            evaluated_models[run_id] = model
            runs_with_models.append(run)
            print(f"Modèle chargé pour run_id: {run_id}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle du run {run_id}: {e}")
            continue # passer au run suivant

        # on recupere les metric
        metrics = run.data.metrics

        # Calcul du score pondéré
        total_score = 0
        all_metrics_present = True

        for metric, weight in metric_weights.items():
            value = metrics.get(metric)
            if value is None:
                all_metrics_present = False
                break  # Ne pas considérer ce run si une métrique est manquante
            total_score += weight * value

        if all_metrics_present and total_score > best_score:
            best_score = total_score
            best_run = run
            best_model = model

    # Enregistrement et transition du meilleur modèle
    if best_run and best_model:
        print(f"🏆 Best model sélectionné : run_id={best_run.info.run_id}, score={best_score:.4f}")
        best_run_id = best_run.info.run_id
        # check model
        if best_model is None:
            raise ValueError(f"❌ Le modèle pour le run sélectionné {best_run_id} n'a pas pu être chargé.")

    
        # Relogger le modèle dans le run sélectionné (optionnel si pas déjà loggé)
        # Enregistrer dans le registry
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(model_uri, "BestModel")
    
        # Promouvoir en production
        client.transition_model_version_stage(
            name="BestModel",
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"Modèle sélectionné : run_id={best_run.info.run_id}, score={best_score:.4f}")
    else:
        print("Aucun run valide avec toutes les métriques n'a été trouvé.")


def reload_best_model():
    #
    url = "http://app/api/v1/reload_model/BestModel"
    #
    headers = {
        "X-Api-Key": API_KEY
    }
    #
    try:
        response = requests.post(url, headers=headers)
        #
        print(response.status_code)
        print(response.json())
    except Exception as e:
        raise RuntimeError(f"Échec reload {model_name} FastAPI : {e}")
    


with DAG(
    dag_id="ml_accidents_pipeline_dag",
    description='Projet MLOps Accident 2025 Datascientest',
    tags=['MLOps', 'datascientest', 'accidents'], 
    default_args=default_args, 
    schedule_interval=None, 
    catchup=False) as dag:
    
    # Task1 Load csv in minio S3
    task1 = PythonOperator(
        task_id="load_csv_to_s3",
        python_callable=load_csv_in_minio,
    )

    # Task2 Prepare Data
    task2 = PythonOperator(
        task_id="prepare_data",
        python_callable=merge_files_by_year,
    )

    # Task3 Clean Data
    task3 = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    task1 >> task2 >> task3
    # 
    reload_tasks = []
    #
    for model_name in models.keys():
        search = PythonOperator(
            task_id=f"search_params_{model_name}",
            python_callable=search_best_params,
            op_kwargs={"model_name": model_name}
        )
        train = PythonOperator(
            task_id=f"train_model_{model_name}",
            python_callable=train_best_model,
            op_kwargs={"model_name": model_name}
        )
        log = PythonOperator(
            task_id=f"log_model_{model_name}",
            python_callable=log_model_mlflow,
            op_kwargs={"model_name": model_name}
        )
        reload = PythonOperator(
            task_id=f"reload_model_{model_name}",
            python_callable=reload_model_api,
            op_kwargs={"model_name": model_name}
        )
        #
        task3 >> search >> train >> log >> reload
        reload_tasks.append(reload)

    select_best = PythonOperator(
        task_id="select_best_model",
        python_callable=select_best_model
    )
#
    reload_best = PythonOperator(
        task_id="reload_best_model",
        python_callable=reload_best_model
    )

    # Définir les dépendances synchronisées
    reload_tasks >> select_best >> reload_best
