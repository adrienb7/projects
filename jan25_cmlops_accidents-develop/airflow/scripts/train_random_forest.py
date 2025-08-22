# train_random_forest.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from minio import Minio
import pandas as pd


TRAIN_BUCKET_NAME = "accidents-train-csv"
MODEL_BUCKET_NAME = "accidents-models"
MINIO_ACCESS_KEY = "XeAMQQjZY2pTcXWfxh4H" 
MINIO_SECRET_ACCESS_KEY = "wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ"

def train_rf(csv_preprocessed_accidents='accidents-preprocessed-csv', model_accidents='accidents-models'):
    #
    minio_client = Minio('minio:9000', access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY, secure=False)
    # S3
    obj = minio_client.get_object(TRAIN_BUCKET_NAME, 'X_train_test.csv')
    X = pd.read_csv(obj, dtype=str, sep=";")
    obj = minio_client.get_object(TRAIN_BUCKET_NAME, 'y_train_test.csv.csv')
    y = pd.read_csv(obj, dtype=str, sep=";").values.ravel()

    # 
    clf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid = GridSearchCV(clf, params, scoring=make_scorer(f1_score, average='macro'), cv=5)
    grid.fit(X, y)

    #joblib.dump(grid.best_estimator_, f'{model_path}/rf_model.pkl')