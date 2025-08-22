import logging
import sys

from pydantic_settings import BaseSettings

from fastapi import APIRouter, Depends, HTTPException, status, Security

# Manage logger
logger = logging.getLogger("api accidents")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler("api.log")
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# Model en cours de test @TODO externaliser dans un json pour le rendre dynamique
MODEL_INFOS = {
    "RandomForest": "Production",
    "SVC": "Production",
    "MLP": "Production",
    "BestModel": "Production"
}

# Class portant les varibale
class Settings(BaseSettings):
    """App settings."""

    project_name: str = "Predict Accidents"
    debug: bool = False
    environment: str = "local"

    # Database
    database_url: str = "postgresql+psycopg2://accidents_user:localhost:8088/mlops_accidents"
    # List API-KEY
    API_KEY: list = ["2eeq-3Ic1-z4Cq-2fdg-0j62-y1B4", "597e-0362-U5ET-870H-fBc0-Z2Fr", "810B-aT2E-465p-097I-fPf6-17Ek","74eC-166E-221S-643Q-9F63-T2FQ","264z-8db0-W1C9-632I-e445-u13o","0c6o-0U62-Q4FP-66dU-089D-W51j","6a78-11dC-E412-0225-2346-V85t","560X-8Ud4-Z601-8b0q-f268-200Q","5bdC-ex76-R73l-9e7W-4t85-f9DL","1c88-3A05-X4DQ-67cP-dL4E-G76g","4f7K-ebc6-g8Bj-5fdy-3FeE-I6Aa","1b96-3ndD-I259-7223-bp25-a00C","7e2D-080C-T1Fb-5b7e-31e2-n64U","46fo-00c2-59Ex-3712-ce07-997g","2fd6-f7fD-208K-4a16-1mb7-20E9","2c1Q-aL78-d00F-86eK-80e7-x1EX","1fc2-4P02-0239-43d3-9w06-T43a","3750-9fc6-473O-0d25-dtb3-Q759","251P-3t8C-G979-23e9-1v39-X60G","0ec1-eJ70-A6CK-879t-8Fb4-H26K","07f6-3I2F-u77U-192t-b24E-E99y","055h-37bD-g5EI-7b2Z-1A67-d3A0","32cB-6N29-c653-782k-ej44-915c","4f83-0936-B7Am-7e06-4H7E-71DB","4dcd-7731-f921-4ccO-bJ2B-37B0","84ae-31aE-U77F-2e4U-0V2E-Y06l","49dz-fw40-N539-4bbe-eHdB-R88g","7aex-5515-L62Q-03bE-7450-n8Fl","82b3-8V96-Y68w-14aH-3sd5-e52Q","6e9A-037C-V9DS-12dZ-8U75-Z8E4","548V-0Wc6-835j-2550-5o01-u37C","778B-csa0-c103-9a5P-9O60-n765","4e9B-fa78-t41e-0964-1g38-f1BB","382W-2Qb0-b4Eo-8082-2u54-U601","683e-ep0D-Z8AN-63cI-f5cE-u9A9","67df-9wc0-W047-1960-2R6C-o17O","5aeG-a716-K15X-1b2r-aAf7-99At","47ej-aN82-N19M-7a3D-a31D-E8C1","8037-2R95-n2Fh-7a9c-5p99-l498","832z-6S30-E14P-9c1r-3Gc1-h66a","30ea-7w7E-a95C-6d2W-ft50-34E1","00e1-co83-m233-1254-0Zd0-85Bz","3c9x-6182-v25Y-03dE-6v32-t30H","1f93-8reC-934M-1cdZ-f08C-o27R","6b21-8f99-n13h-252m-4T29-s5F3","1c2n-236E-12BB-949b-7OcD-469w","7bex-4bbA-V0Bz-851o-eL4D-Y8FO","6c32-cQ49-z859-6527-b935-81A7","284P-d5b7-a88k-1e1c-9L26-u8Br","0d4f-2D14-I50o-0288-fC08-i327"]
    #API_KEY: str = "2eeq-3Ic1-z4Cq-2fdg-0j62-y1B4"
    API_KEY_NAME: str = "X-Api-Key"

    #
    logger: object = logger

    #


#
settings = Settings()



