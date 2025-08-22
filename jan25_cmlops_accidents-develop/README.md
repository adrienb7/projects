**PROJET DE SOUTENANCE DU MODULE DATASCIENTEST MLOPS JANVIER 2025**
================================================================

  **Accidents routiers en France**
=========================================

# Description du projet :
- L’objectif de ce projet est d’essayer de prédire la gravité des accidents routiers en France.
- Les prédictions seront basées sur les données historiques.
- C’est un problème parfait pour traiter l’ensemble des étapes d’un projet de Data Science. 
- Une première étape est d’étudier et appliquer des méthodes pour nettoyer le jeu de données. Une fois le jeu de données propre, une deuxième étape est d’extraire à partir de l'historique des matchs les caractéristiques qui semblent être pertinentes pour estimer la gravité d’accidents. 
- Ensuite, à partir de ses résultats, l’objectif est de travailler un scoring des zones à risque en fonction des informations météorologiques, l’emplacement géographique (coordonnées GPS, images satellite, etc …)
- Les clients sont les policiers ou le SAMU qui souhaitent avoir une estimation de l’urgence en temps réel pour dimensionner l’intervention.
- Aspects spécifiques du projet :
  - Etablir un fonctionnement optimal du modèle
  - Comment garantir une rapidité d’exécution suffisante ?

# Ressources à consulter :
- Données :
  - https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/
  - https://www.kaggle.com/ahmedlahlou/accidents-in-france-from-2005-to-2016

- Bibliographie :
  - Bases de données annuelles des accidents corporels de la circulation routière - Années de 2005 à 2020 - data.gouv.fr
- Template Github :
  - https://github.com/DataScientest-Studio/Template_MLOps_accidents.git

# Présentation de la soutenances
- [Slide de la présentation](https://docs.google.com/presentation/d/1bRGIvbZdinHLpCwq-0VENwzRAdNXl4Xfc2ZStPCwVQY/edit?usp=sharing)
- [Github du projet](https://github.com/DataScientest-Studio/jan25_cmlops_accidents/tree/develop)


# Structure du projet

```
./jan25_cmlops_accidents
├── accidents.log
├── airflow
│   ├── dags
│   │   ├── dag_accidents.py
│   │   ├── dag_train_with_mlflow.py
│   │   ├── first_dag.py
│   │   └── __pycache__
│   ├── Dockerfile
│   ├── logs
│   │   ├── dag_id=accidents_training_pipeline_api
│   │   ├── dag_id=ml_accidents_pipeline_dag
│   │   ├── dag_id=mlops_2025_accidents_dag
│   │   ├── dag_processor_manager
│   │   └── scheduler
│   ├── plugins
│   └── scripts
│       ├── __init__.py
│       └── train_random_forest.py
├── data
│   ├── preprocessed
│   │   ├── accidents graves proportions heatmap.png
│   │   ├── df_accident_clean.csv
│   │   ├── df_accidents_2019.csv
│   │   ├── df_accidents_2020.csv
│   │   ├── df_accidents_2021.csv
│   │   ├── df_accidents_2022.csv
│   │   ├── df_accidents_2023.csv
│   │   ├── df_accidents_clean.csv
│   │   └── df_accidents.csv
│   └── raw
│       ├── caracteristiques-2019.csv
│       ├── caracteristiques-2020.csv
│       ├── caracteristiques-2021.csv
│       ├── caracteristiques-2022.csv
│       ├── caracteristiques-2023.csv
│       ├── lieux-2019.csv
│       ├── lieux-2020.csv
│       ├── lieux-2021.csv
│       ├── lieux-2022.csv
│       ├── lieux-2023.csv
│       ├── usagers-2019.csv
│       ├── usagers-2020.csv
│       ├── usagers-2021.csv
│       ├── usagers-2022.csv
│       ├── usagers-2023.csv
│       ├── vehicules-2019.csv
│       ├── vehicules-2020.csv
│       ├── vehicules-2021.csv
│       ├── vehicules-2022.csv
│       └── vehicules-2023.csv
├── db
│   ├── conf
│   │   └── postgresql.conf
│   ├── Dockerfile
│   └── sql
│       ├── 00_init_mlops_accidents.sql
│       ├── 01_init_mflow_db.sql
│       ├── 03_init_dataset_structure.sql
│       └── 04_init_airflow_db.sql
├── docker-compose.yaml
├── fastapi
│   ├── api.log
│   ├── app
│   │   ├── api
│   │   ├── api.log
│   │   ├── auth.py
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── mlruns
│   │   └── __pycache__
│   ├── curl.txt
│   ├── Dockerfile
│   ├── requirements.txt
│   └── venv_api
├── LICENSE
├── mlflow
│   ├── arch_mlflow_minio_posrgresql.png
│   └── Dockerfile.mlflow
├── mlruns
├── models
│   ├── accident_severity_model_tabnet.pkl
│   └── accident_severity_model_xgboost_gpu.pkl
├── notebooks
│   └── preparation_donnees_v1.ipynb
├── postman
│   └── accidents_datascientest.postman_collection.json
├── README_AIRFLOW.md
├── README.md
├── references
├── reports
│   └── figures
├── requirements.txt
├── schema
│   ├── accidents_worflow.png
│   ├── airflow_tasks.png
│   ├── DAG_AIRFLOW.drawio
│   ├── DAG_AIRFLOW.png
│   ├── DD_mlops_accidents.png
│   ├── dictionnaire_models.png
│   ├── docker_compose_diagram.png
│   ├── ER_DATASET.drawio
│   ├── ET_AIRFLOW.drawio
│   ├── ET_AIRFLOW.png
│   └── taks_airflow_dynamique_traitement_model_ML.png
├── scripts
│   ├── csv_to_postgres.py
│   ├── diagramme_composant.py
│   ├── first_dag.py
│   ├── generate_curl_from_line.py
│   ├── generate_curls_from_csv.py
│   ├── init_db_accidents.sh
│   ├── launch_stack_docker.sh
│   ├── README_INSTALL_DOCKER_LINUX.md
│   └── test_csv_to_postgres.py
├── setup_env.sh
├── src
│   ├── data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── db
│   │   ├── database.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── repositoryDatasetFeatures.py
│   │   └── service.py
│   ├── features
│   │   ├── build_features.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── mlruns
│   │   ├── 0
│   │   ├── 194121514032077162
│   │   ├── 212788986025115102
│   │   ├── 379032889499689169
│   │   ├── 472813414145023489
│   │   └── models
│   ├── models
│   │   ├── display_xgboost_mlflow.py
│   │   ├── __init__.py
│   │   ├── predict_model.py
│   │   ├── randomforestregressor_mlflow_test.py
│   │   ├── train_logisticRegression.py
│   │   ├── train_model.py
│   │   ├── train_random_classifier_cpu.py
│   │   ├── train_tabnet.py
│   │   └── train_xgboost.py
│   ├
│   ├── test
│   │   ├── db
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── test_pandas_float.py
│   └── visualization
│       ├── accidents graves proportions heatmap.png
│       ├── heatmap_proportion3_gravite_20000.html
│       ├── __init__.py
│       └── visualize.py
└── venv
    ├── bin
```

# Installation du projet

## Clonner le projet

- git clone --branch develop --single-branch https://github.com/DataScientest-Studio/jan25_cmlops_accidents.git

## Environement Docker

- Creation du fichier .env

```bash
#POSTGRESQL
PG_USER=postgres
PG_PASSWORD=Ws4E8yP7C2coI0UA
PG_PORT=5432

# MLOP ACCIDENTS DB
PG_DB_ACCIDENTS_NAME=mlops_accidents
PG_USER_ACCIDENTS=accidents_user
PG_PASSWORD_ACIDENTS=zcb8TXWa2bkX

# MLFLOW 
PG_DB_MLFLOW_NAME=mlflow
PG_USER_MLFLOW=mlflow_user
PG_PASSWORD_MLFLOW=zcb8TXWa2bkY
MLFLOW_PORT=5000
                   
# MINIO
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
# il faut créer les clés sur le GUI
MINIO_ACCESS_KEY=XeAMQQjZY2pTcXWfxh4H 
MINIO_SECRET_ACCESS_KEY=wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ
#NINIO_REGION=us-east-1
MINIO_BUCKET_NAME=mlflow
MINIO_API_ADDRESS='0.0.0.0:9000' 
MINIO_STORAGE_USE_HTTPS=False 
MINIO_CONSOLE_ADDRESS='0.0.0.0:9001' 
MINIO_API_PORT=9000 
MINIO_CONSOLE_PORT=9001

# FAST API
EXT_FAST_API_PORT=8089
LOCAL_FAST_API_PORT=80

# AIRFLOW
AIRFLOW_UID=1000
AIRFLOW_GID=0
AIRFLOW__CORE__LOAD_EXAMPLES=False # ne pas faire apparitre les exemples

```
- networks 
	 - back-tier  : 172.21.0.0/16
	 - front-tier : 172.22.0.0/16
	 - La dns interne de docker permet de cloissoner les networks. 
	 - Les containers pocédant les 2 networks peuvent communiquer avec la valeur de la varibale hostname (declaré dans le service)


## Lancer la stack docker

```bash
/jan25_cmlops_accidentsdocker-compose/scripts/launch_stack_docker.sh

## OU

./jan25_cmlops_accidentsdocker-compose up -d
```
## MINIO PARAMETRAGE
- Lancer le navigateur avec l'url http://localhost:9001/
- login    : minio
- password : minio123

### Creation de l'Acces Key
- Menu "Access Keys"
  - Ceation de l'acces key : **XeAMQQjZY2pTcXWfxh4H**

### Creation des buckets S3:// (MINIO)
- Creation des buckets :
  - accidents-raw-csv 
  - accidents-preprocessed-csv
  - accidents-train-csv
  - accidents-models
  - mlflow

## Lancement du dag Airflow

- url :
  - http://localhost:8080/login/
  - login    : airflow
  - password : airflow
- lancement du dag : **ml_accidents_pipeline_dag**
  - Lancer le dag plusieur fois pour avoir plusieur run dans "l'Experiments : **Gravite_Accidents**"
- [README_AIRFLOW_ACCIDENTS.md](README_AIRFLOW_ACCIDENTS.md)

## Api Rest Fastapi
- [README_FASTAPI_ACCIDENTS.md](README_FASTAPI_ACCIDENTS.md)

## Collection Postman du test
- dans le répertoire ./postman nous pouvons importer la collection de test et sont environnement
- [README_POSTMAN_ACCIDENTS.md](README_POSTMAN_ACCIDENTS.md)

## Le Docker-compose.yaml
- [README_DOCKERCOMPOSE_ACCIDENT.md](README_DOCKERCOMPOSE_ACCIDENT.md)


## Les Urls
- airflow  : http://localhost:8080/
- Mlflow   : http://localhost:5000/
- Minio    : http://localhost:9001/
- FastApi  : http://localhost:8089/api/docs

=====================================================================================

# Utilisation des sources en dehors de la stack doker
- repartoiore ./src

## Vitual Environement et PYTHONPATH

- Pour les dépendances entre les packages/modules que l'on développent, il est nécessaire de définir correctemet lla Variable PYTHONPATH.
- Dans le cadre de notre projets nous avons les sources qui se trouvent dans le répertoire src donc :

```bash
export PYTHONPATH=src
```

- Il faut lancer le virtuel environement dans un script **'setup_env.sh'** bash qui définie a variable PYTHONPATH :

```bash
#! bin/bash

export PYTHONPATH=src

source venv/bin/activate

```

# ANNEXES
 pocédant les 2 networks peuvent communiquer avec la valeur de la varibale hostname (declaré dans le service)

- postgresql
- [postgresql.conf](./db.conf/postgresql.conf)
  - instance pour stoker les csv **pas utilisé**
    - hostname : postgresql_accidents
    - ports    : 5432 exposé que sur le network back-tier
    - base     : mlops_accidents
    - user     : accidents_user
    - password : zcb8TXWa2bkX
  - instance pour mlflow
    - base     : mlflow
    - password : zcb8TXWa2bkY
  - Avant de lancer la stack docker, avec le script "/script/lauch_stack_docker.sh, il faut créer le fichier '.env' pour le docker-compose.yaml

- Le Dockerfile qui se trouve dans le repetoire db, copie le script 00_init_mlops_accidents.sql, 01_init_mlflow_db.sql, db/sql/03_init_dataset_structure.sql et db/sql/04_init_airflow_db.sql dans le repertoire de l'image /docker-entrypoint-initdb.d/ 
- Lors de la creation de l'image le containers execute le script sql dans l'orde 01, 02, 03, 04, après la création de l'instance posgresql:
  - 00_init_mlops_accidents.sql.  : Il créer le user accident_user et la base mlops_accidents.[00_init_mlops_accidents.sql](./db/sql/00_init_mlops_accidents.sql)

  - 01_init_mlflow_db.sql         : Il créer le user mlflow_user et l'instance pour la base mlflow [01_init_mflow_db.sql](01_init_mflow_db.sql)

  - 03_init_dataset_structure.sql : Il créer, les relation de la base mlops_accidents [03_init_dataset_structure.sql](03_init_dataset_structure.sql)
   
  - 04_init_airflow_db.sql        :  Il créer la base et le user de la base airflow [04_init_airflow_db.sql](04_init_airflow_db.sql)
 
 - pour réinitailisé (vider les bases de données) il faut supprimer lesz volume persistant du service posgresql.
 - pour cela vous pouvez utiliser le script [./scripts/init_db_accidents.sh](./scripts/init_db_accidents.sh) !! supprime toutes les instances !!