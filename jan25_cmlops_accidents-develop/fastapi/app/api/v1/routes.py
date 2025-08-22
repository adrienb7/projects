from fastapi import APIRouter, Depends, Request, HTTPException, Query
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import mlflow
from auth import get_api_key
from config import settings, MODEL_INFOS
from api.models import PredictAccidentsPayload
from typing import Optional
import pandas as pd

#
router = APIRouter()

models = {}# modèle par nom
model_versions = {}  # version du modèle par nom

#
def load_model(model_name: str, version: str):
    #uri = f"models:/{model_name}/{MODEL_INFOS[model_name]}"
    uri = f"models:/{model_name}/{version}"
    return mlflow.pyfunc.load_model(uri)


def get_latest_model_version(model_name: str) -> str:
    print("MLFLOW URI =", mlflow.get_tracking_uri())
    try:
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        settings.logger.info(f"=> versions : {versions}")
        versions = list(versions)  # Force l'itération, car c'est un generator-like
        if not versions:
            raise HTTPException(status_code=404, detail=f"Aucune version trouvée pour le modèle '{model_name}'")
        # On trie les versions par numéro décroissant
        latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        settings.logger.info(f"=> versions : {latest_version}")
        return latest_version.version
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des versions : {e}")

# API TEST
@router.get("/test")
async def test():
    settings.logger.info("=> TEST API OK ...")
    return {"detail": "API FONCTIONNELLE"}

@router.get("/test/{echo}")
async def secure_test(
    echo: str,  # 
    api_key_header: settings.API_KEY_NAME = Depends(get_api_key)
):
    settings.logger.info(f"=> TEST API {echo} OK ...")
    return {"detail": f"API FONCTIONNELLE {echo}"}
    
#
"""
    Description:
    Cette route renvoie un message uniquement si une clé API valide est fournie dans l'en-tête.

    Args:
    - api_key_header (APIKey, dépendance): La clé API fournie dans l'en-tête de la requête.

    Returns:
    - si la clé API est valide.

    Raises:
    - HTTPException(401, detail="Unauthorized"): Si la clé API est invalide ou manquante, une exception HTTP 401 Unauthorized est levée.
    """
@router.post("/predict/{model_name}")
#async def secure(request: Request, api_key_header: settings.API_KEY_NAME = Depends(get_api_key)):
async def secure_predict(
    model_name: str,  # 👈 d'abord les paramètres simples (path, query)
    payload: PredictAccidentsPayload,  # 👈 ensuite le corps de la requête
    api_key_header: settings.API_KEY_NAME = Depends(get_api_key),  # 👈 puis les dépendances
):
    # validate payload
    payload = payload.model_validate(payload)
    settings.logger.info(f"=> payload {payload}-{model_name}")
    print(f"=> payload {payload}-{model_name}")
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"❌ Modèle '{model_name}' non chargé. Utilisez /reload_model pour le charger.")
    #    
    model = models[model_name]
    #
    try:
        # Adaptation pour prédire avec un DataFrame
        input_df = pd.DataFrame([payload.model_dump()])
        prediction = model.predict(input_df)
        version_used = model_versions.get(model_name, "unknown")
        return {"model": model_name, "prediction": prediction.tolist(), "version": version_used}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur durant la prédiction : {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur durant la prédiction : {e}")

# Reload dynamique d'un modele
@router.post("/reload_model/{model_name}")
async def secure_reload_model(
    model_name: str, 
    version: Optional[str] = Query(None),  # 👈 d'abord les paramètres simples (path, query)
    api_key_header: settings.API_KEY_NAME = Depends(get_api_key),  # 👈 puis les dépendances
):
    print(f"=> model_name : {model_name}, version : {version}")
    try:
        version_to_load = version or get_latest_model_version(model_name)
        uri = f"models:/{model_name}/{version_to_load}"
        model = mlflow.pyfunc.load_model(uri)
        models[model_name] = model
        model_versions[model_name] = version_to_load  # 👈 on sauvegarde la version ici
        return {"message": f"✅ Modèle '{model_name}' version '{version_to_load}' rechargé avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur : {e}")