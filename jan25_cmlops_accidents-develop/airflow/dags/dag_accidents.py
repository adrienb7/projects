from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from minio import Minio
from io import BytesIO
import pandas as pd
from datetime import datetime
import os

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


RAW_BUCKET_NAME = "accidents-raw-csv"
PREPROCESSED_BUCKET_NAME = "accidents-preprocessed-csv"
TRAIN_BUCKET_NAME = "accidents-train-csv"

LOCAL_DIR = '/app/data/raw/'
LOCAL_DIR_PREPROCESSED = '/app/data/preprocessed/'


# Put csv in Minio
def load_csv_in_minio():
    try:
        # df_c_2019 = pd.read_csv("/app/data/raw/caracteristiques-2019.csv", dtype=str, sep=";")
        #source_file = "/app/data/raw/caracteristiques-2019.csv"
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
            print(f"Downloading {file} from MinIO...")
            #minio_client.fput_object(RAW_BUCKET_NAME, file, f"{LOCAL_DIR}{file}")  # Récupérer le fichier en mémoire
            obj = minio_client.get_object(RAW_BUCKET_NAME, file)
            #
            df = pd.read_csv(obj, dtype=str, sep=";")
            #df = pd.read_csv(BytesIO(data.read()), on_bad_lines='warn')  # Charger en DataFrame
            #df = pd.read_csv(f"{LOCAL_DIR}{file}", dtype=str, sep=";")
            
            if 'lieux' in file:
                dfs['lieux'] = df
            elif 'vehicules' in file:
                dfs['vehicules'] = df
            elif 'usagers' in file:
                dfs['usagers'] = df
            elif 'caracteristiques' in file:
                print(f"{year}")
                if '2022' in year:
                    dfs['caracteristiques'] = df.rename(columns={"Accident_Id": "Num_Acc"}, inplace=True)
                dfs['caracteristiques'] = df

            print(f"=> {df.head()}")

            # Une fois le fichier traité, le supprimer de MinIO
            #minio_client.remove_object(RAW_BUCKET_NAME, file)
            #print(f"File {file} deleted from MinIO.")

        # Fusion
        merged_df = dfs['caracteristiques'].merge(dfs['lieux'], on='Num_Acc').merge(dfs['vehicules'], on='Num_Acc').merge(dfs['usagers'], on=['Num_Acc', 'id_vehicule'])
        #merged_df = pd.merge(dfs['lieux'], dfs['vehicules'], on='Num_Acc')
        #merged_df = pd.merge(merged_df, dfs['caracteristiques'], on='Num_Acc')
        #merged_df = pd.merge(merged_df, dfs['usagers'], on=['Num_Acc', 'id_vehicule'])
        # Sauve in minio processed
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
                       'actp', 'etatp', 'id_usager']
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

    # Feature && target
    X = data.drop(['grav'], axis=1).astype('float')
    y = data['grav']

    # Division de notre dataset en deux. Pour la cross validation (train_test) 90 % des données et 10% des données pour tester les modèles finaux (global_test)
    X_train_test, X_global_test, y_train_test, y_global_test = train_test_split(X, y, test_size=0.10)

    # Sauvegarder clean data
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'df_accident_clean.csv')
    data.to_csv(output_file, index=False)
    minio_client.fput_object(PREPROCESSED_BUCKET_NAME, 'df_accident_clean.csv', output_file)

    # Sauvegarder les splits train's
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'X_train_test.csv')
    data.to_csv(output_file, index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'X_train_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}X_train_test.csv")
    #
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'X_global_test.csv')
    data.to_csv(output_file, index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'X_global_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}X_global_test.csv")
    #
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'y_train_test.csv')
    data.to_csv(output_file, index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'y_train_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}y_train_test.csv")
    #
    output_file = os.path.join(LOCAL_DIR_PREPROCESSED, 'y_global_test.csv')
    data.to_csv(output_file, index=False)
    minio_client.fput_object(TRAIN_BUCKET_NAME, 'y_global_test.csv', output_file)
    os.remove(f"{LOCAL_DIR_PREPROCESSED}y_global_test.csv")

     

    ## Dict (tableau associatif) avec le nom du model et l'objet methode lui correspondant
    #model_classes = {
    #    'LinearRegression': LinearRegression,
    #    'DecisionTreeRegressor': DecisionTreeRegressor,
    #    'RandomForestRegressor': RandomForestRegressor
    #}
#
    ## Instanciation Objet de la fonction
    #model_class = model_classes[model]
#
    ## computing cross val
    #cross_validation = cross_val_score(
    #    model_class(),
    #    X,
    #    y,
    #    cv=3,
    #    scoring='neg_mean_squared_error')
#
    #model_score = cross_validation.mean()
#
    ## transmition du score via mécanisme Xcom (Task-id: nom de la tâche), key:value)
    #kwargs['ti'].xcom_push(key="model_score", value=model_score)  # Push to XCom



# Dag definition
with DAG(
    dag_id='mlops_2025_accidents_dag',
    description='',
    tags=['MLOps', 'datascientest', 'accidents'],
    schedule_interval=None,
    # schedule_interval="* * * * *",  # chaque minute
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2025, 5, 15, 0, 0),  # date fixe dans le passé
    },
    catchup=False  # ne pas rejouer les minutes passées
) as accidents_dag:
    
    # Task1 Load csv in minio S3
    task1 = PythonOperator(
        task_id="load_csv_to_s3",
        python_callable=load_csv_in_minio,
        # op_kwargs: Optional[Dict] = None,
        # op_args: Optional[List] = None,
        # templates_dict: Optional[Dict] = None
        # templates_exts: Optional[List] = None
    )

    # Task2 Prepare Data
    task2 = PythonOperator(
        task_id="prepare_data",
        python_callable=merge_files_by_year,
        # op_kwargs: Optional[Dict] = None,
        # op_args: Optional[List] = None,
        # templates_dict: Optional[Dict] = None
        # templates_exts: Optional[List] = None
    )

    # Task3 Clean Data
    task3 = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
        # op_kwargs: Optional[Dict] = None,
        # op_args: Optional[List] = None,
        # templates_dict: Optional[Dict] = None
        # templates_exts: Optional[List] = None
    )

    # Task Group 4 compute models score
    

    # Task 5 tarin and save baset model

    # Définition de l'ordre des tâches (routage)
    task1 >> task2 >> task3