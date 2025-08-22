import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Paramètres PostgreSQL 
DB_USER = "accidents_user"
DB_PASSWORD = quote_plus("zcb8TXWa2bkX")
DB_HOST = "postgresql_accidents"
DB_PORT = "5432"
DB_NAME = "mlops_accidents"

# Chargement du dataset "brut" (accidents.csv)
df = pd.read_csv("accidents.csv", encoding="utf-8")
print(f"accidents.csv chargé ({len(df)} lignes)")

#  Étape 1 : Créer le contenu de te_datasets 
df_datasets = pd.DataFrame([{
    "id": 1,
    "description": "Données accidents 2019-2023 nettoyées V1",
    "creation_dt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modification_dt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "is_new": True,
    "code": "ACC-2019-2023",
    "type": "accidents"
}])

# Étape 2 : Créer le contenu de te_features 
features = df.columns.tolist()
df_features = pd.DataFrame({
    "id": range(1, len(features)+1),
    "code": features
})

# Étape 3 : Créer la table de relation ta_dataset_features 
df_link = pd.DataFrame({
    "dataset_id": 1,
    "feature_id": range(1, len(features)+1)
})

# Connexion PostgreSQL 
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
print("Connexion PostgreSQL établie")

# Import dans la base 
df_datasets.to_sql("te_datasets", engine, if_exists="append", index=False)
print("Table te_datasets remplie")

df_features.to_sql("te_features", engine, if_exists="append", index=False)
print("Table te_features remplie")

df_link.to_sql("ta_dataset_features", engine, if_exists="append", index=False)
print("Table ta_dataset_features remplie")


# Fermeture de la connexion 
engine.dispose()
print("Connexion fermée - Import terminé")
