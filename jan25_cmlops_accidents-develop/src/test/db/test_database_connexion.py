import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from datetime import datetime
import random 
# Own Modules
from db.models import DataSet, Feature, DatasetFeature  # Assurez-vous que DataSet est le modèle que vous souhaitez tester
#from db.database import SessionLocal, engine  # Le fichier où votre code de base de données est défini

# conf
# Paramètres PostgreSQL @TODO à rendre dynamique avec un .env
DB_USER = "accidents_user"
DB_PASSWORD = quote_plus("zcb8TXWa2bkX")
# DB_HOST = "postgresql_accidents"
DB_HOST = "0.0.0.0"
DB_PORT = "5435"
DB_NAME = "mlops_accidents"

# url de connexion
SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Connexion à PostgreSQL
engine_test = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocalTest = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

# Recréer les tables pour chaque test (si nécessaire)
@pytest.fixture(scope="module")
def setup_database():
    """Fixture pour configurer la base de données avant les tests et nettoyer après."""
    # Créer les tables dans la base de données de test (en utilisant le modèle DataSet)
    #Base.metadata.create_all(bind=engine_test)
    
    # Créer une session de base de données pour l'exécution des tests
    db = SessionLocalTest()
    
    #daset_code
    #dataset_code = f"DS-{str(random.randint(0,100))}"
    #print(f"=> code : {dataset_code}")
#
    ## Insérer des données de test (par exemple, un DataSet de test)
    #db.add(DataSet(code=dataset_code, creation_dt=datetime.now(), description='TEST', type='TRAIN'))  # Exemple de DataSet
    #db.commit()
    yield db  # Ceci permet d'utiliser la base de données pendant les tests

    #
    #print(f"=> Display all records : {db.query(DataSet).all()}")
    # Nettoyer la base de données après les tests
    #db.query(DataSet).delete()
    #db.close()

# Test de la connexion et de la session
def test_database_connection(setup_database):
    """Test si la connexion à la base de données fonctionne et qu'une session est créée correctement."""
    db = setup_database
    assert db is not None  # Vérifier si la session est bien créée

    # Delete all entity in database
    # begin by association table
    list_dataset_features = db.query(DatasetFeature).all()
    if list_dataset_features is not None:
        for dataset_feature in list_dataset_features:
            db.delete(dataset_feature)
            db.commit()

    
    # Delete all Dataset
    list_dataset = db.query(DataSet).all()
    if list_dataset is not None:
        for dataset in list_dataset:
            db.delete(dataset)
            db.commit()
    
    # Delete all features
    list_features = db.query(Feature).all()
    if list_features is not None:
        for feature in list_features:
            db.delete(feature)
            db.commit()

    #
    assert db.query(DataSet).count() == 0  # Vérifier qu'un dataset a été inséré


# Test d'insertion et de lecture dans la base de données
def test_insert_and_read_dataset(setup_database):
    """Test de l'insertion d'un DataSet et de la lecture depuis la base de données."""
    db = setup_database
    # Insérer un dataset
    # DataSet(code=f"DS001-{datetime.now()}"[0-14], creation_dt=datetime.now(), description='TEST', type='TRAIN')
    dataset_code = f"DS-{str(random.randint(0,100))}"    
    print(f"=> code : {dataset_code}")       
    #        
    new_dataset = DataSet(code=dataset_code, creation_dt=datetime.now(), description='TEST', type='TRAIN')
    #
    db.add(new_dataset)
    db.commit()

    # Display Id
    print(f"=> After read id : {new_dataset.id}")
    
    # Lire le dataset inséré
    inserted_dataset = db.query(DataSet).filter(DataSet.code == dataset_code).first()

    # Display Before Read
    print(f"=> Before read id : {new_dataset.id}")
    
    assert inserted_dataset is not None  # Vérifier que l'élément a bien été inséré
    assert inserted_dataset.code == dataset_code  # Vérifier que le code du dataset est correct

    # delete dataset
    db.delete(inserted_dataset)
    db.commit()

# Test de suppression dans la base de 
def test_delete_dataset(setup_database):
    """Test de la suppression d'un DataSet dans la base de données."""
    db = setup_database
    
    # - dataset_code
    dataset_code = f"DS-{str(random.randint(0,100))}"                   
    
    # Insérer un dataset à supprimer
    dataset_to_delete = DataSet(code=dataset_code, creation_dt=datetime.now(), description='TEST', type='TRAIN')
    db.add(dataset_to_delete)
    db.commit()

    # Vérifier que le dataset est inséré
    dataset = db.query(DataSet).filter(DataSet.code == dataset_code).first()
    assert dataset is not None
    if dataset is not None:
        print(f"=> dataset : {dataset}")

    # Supprimer le dataset
    db.delete(dataset)
    db.commit()

    # Vérifier que le dataset a bien été supprimé
    dataset = db.query(DataSet).filter(DataSet.code == dataset_code).first()
    assert dataset is None


# Test Create datset with features
def test_create_dataset_features(setup_database):
    # Acquire session
    db = setup_database

    # Create Dataset
    new_dataset_1 = DataSet(code=f"DS-{str(random.randint(0,100))}", creation_dt=datetime.now(), description='TEST', type='TRAIN')
    new_dataset_2 = DataSet(code=f"DS-{str(random.randint(0,100))}", creation_dt=datetime.now(), description='TEST', type='TRAIN')
    tab_dataset = [new_dataset_1, new_dataset_2]

   
    # add entity dataset
    db.add_all(tab_dataset)

    # Create features []
    tab_features = [Feature(code="mois"), Feature(code="aa"), Feature(code="hhmm"), Feature(code="vit")]
    # Add enttity Feature
    db.add_all(tab_features)

    # First Commit
    db.commit()
    
    # Récupération des IDs
    # Récupération des identifiants (ID) des objets créés
    dataset_ids = [dataset.id for dataset in tab_dataset]
    feature_ids = [feature.id for feature in tab_features]

    # Create association
    list_datastet_features = []
    for dataset_id in dataset_ids:
        for feature_id in feature_ids:
            print(f"=> dataset_id : {dataset_id} : feature_id : {feature_id}")
            list_datastet_features.append(DatasetFeature(dataset_id=dataset_id, feature_id=feature_id, value=random.randint(0,100)))
    
    
    # Next Transaction
    #db.begin()


    # Insertion link dataste features
    db.add_all(list_datastet_features)

    # Second commit
    db.commit()

    # release connection
    db.close()

## Test pour vérifier l'échec de la connexion avec une mauvaise URL
#@pytest.fixture
#def test_invalid_connection():
#    """Test de la connexion avec des paramètres incorrects"""
#    try:
#        # Tentative de connexion avec une mauvaise URL
#        invalid_engine = create_engine("postgresql://wrong_user:wrong_password@localhost:5432/invalid_db")
#        SessionLocalInvalid = sessionmaker(autocommit=False, autoflush=False, bind=invalid_engine)
#        db_invalid = SessionLocalInvalid()
#        db_invalid.close()
#    except Exception as e:
#        assert isinstance(e, Exception)  # Vérifier que l'exception se produit correctement
#

