import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from pathlib import Path
import sys
import os
from datetime import datetime

# conf
# Paramètres PostgreSQL 
DB_USER = "accidents_user"
DB_PASSWORD = quote_plus("zcb8TXWa2bkX")
#DB_HOST = "postgresql_accidents"
DB_HOST = "0.0.0.0"
DB_PORT = "5435"
DB_NAME = "mlops_accidents"

# Global
project_dir = Path(__file__).resolve().parents[2]
print(f"=> Root Path : {project_dir}")

# Connexion à PostgreSQL
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
# engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')

# Charger les fichiers CSV dans des DataFrames Pandas
df_accidents = pd.read_csv(f"{project_dir}/jan25_cmlops_accidents/data/preprocessed/df_accidents.csv", low_memory=False)
print(f"=> load df_accidents {df_accidents.head(5)}")
print(f"=> load df_accidents.columns {df_accidents.columns}")
#
df_accident_test =  pd.DataFrame(df_accidents.head(5), columns=df_accidents.columns)
print(f"=> load df_accidents_test {df_accident_test}"
      
      )
# preparation df_dataset, df_features et df_link

timestamp = datetime.now()

# new dataset
dataset_dic = {
  "description": ["Test Creation DataSet"],
  "creation_dt": [timestamp],
  "modification_dt": [timestamp],
  "is_new": [True],
  "code": ["DATASET_TEST"],
  "type": ["TRAIN"]
}
df_dataset = pd.DataFrame(dataset_dic)
print(f"=> df_dataset : {df_dataset.head()}")
# Insertion dataset 
df_dataset.to_sql(name='te_datasets', con=engine,schema='public', if_exists='append', index=False)
# Recupération des ids
print(f"=> dataset code {dataset_dic['code'][0]}")
dataset_query_id = f"SELECT id FROM te_datasets WHERE code = '{dataset_dic['code'][0]}';"
print(f"=> query : {dataset_query_id}")
df_dataset_id = pd.read_sql(dataset_query_id, engine)

# features
df_features = pd.DataFrame(columns=df_accident_test.columns)
df_features.to_sql('te_features', engine, if_exists='append', index=False)
# Recupération des id (df['Nom'].tolist())
#                   f"SELECT * FROM users WHERE user_name IN   ({', '.join(f"'{user_id}'" for user_id in user_ids)})"
feature_list = df_features.columns.tolist()
formatted_features = ', '.join([f"'{item}'" for item in feature_list])
features_query_id = f"SELECT id FROM te_features WHERE code IN ({formatted_features})"
df_features_id = pd.read_sql(features_query_id, engine)

print(f"=> {df_dataset}")
print(f"=> {df_features}")
print(f"=> {df_features_id}")

# 
#df_entity1 = pd.read_csv('entity1.csv')
#df_entity2 = pd.read_csv('entity2.csv')
#df_association = pd.read_csv('association.csv')
#df.head(5), columns=df.columns
# Insérer les données dans PostgreSQL
# Insérer les données dans les tables d'entités
#df_entity1.to_sql('entity1', engine, if_exists='replace', index=False)
#df_entity2.to_sql('entity2', engine, if_exists='replace', index=False)

# Récupérer les IDs générés pour les entités
##entity1_ids_query = "SELECT id, name FROM entity1;"
##entity2_ids_query = "SELECT id, description FROM entity2;"
##
##df_entity1_ids = pd.read_sql(entity1_ids_query, engine)
##df_entity2_ids = pd.read_sql(entity2_ids_query, engine)
##
### Afficher les IDs récupérés
##print(df_entity1_ids)
##print(df_entity2_ids)
##
### Mettre à jour le DataFrame d'association avec les IDs
##df_association = df_association.merge(df_entity1_ids, left_on='entity1_name', right_on='name', how='left')
##df_association = df_association.merge(df_entity2_ids, left_on='entity2_desc', right_on='description', how='left')
##
### Mettre à jour la table d'association avec les IDs
##df_association = df_association[['id_x', 'id_y']]
##df_association.columns = ['entity1_id', 'entity2_id']
##
### Insérer les données dans la table d'association
##df_association.to_sql('association_table', engine, if_exists='replace', index=False)
##
### Vérifier l'insertion dans la table d'association
##association_query = "SELECT * FROM association_table;"
##df_association_result = pd.read_sql(association_query, engine)
##print(df_association_result)
##