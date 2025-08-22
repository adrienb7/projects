import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from sqlalchemy.exc import SQLAlchemyError

# conf
# Paramètres PostgreSQL @TODO à rendre dynamique avec un .env
DB_USER = "accidents_user"
DB_PASSWORD = quote_plus("zcb8TXWa2bkX")
# DB_HOST = "postgresql_accidents"
DB_HOST = "0.0.0.0"
DB_PORT = "5435"
DB_NAME = "mlops_accidents"


class Database:
    def __init__(self):
        # Récupération des variables d'environnement pour la connexion à la base de données
        self.db_host = os.getenv('DB_HOST', DB_HOST)
        self.db_port = os.getenv('DB_PORT', DB_PORT)
        self.db_name = os.getenv('DB_NAME', DB_NAME)
        self.db_user = os.getenv('DB_USER', DB_USER)
        self.db_password = os.getenv('DB_PASSWORD', DB_PASSWORD)

        # Création du moteur de base de données
        self.engine = self.create_engine()

        # Création de la session
        self.Session = self.create_session()

    def create_engine(self):
        """
        Crée le moteur de base de données PostgreSQL.
        """
        try:
            engine = create_engine(
                f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}',
                pool_size=10,
                max_overflow=20,
                pool_recycle=300,
                #connect_args={"check_same_thread": False},
                echo=True
            )
            return engine
        except SQLAlchemyError as e:
            print(f"Erreur lors de la création du moteur de base de données : {e}")
            raise e

    def create_session(self):
        """
        Crée une session de base de données.
        """
        try:
            Session = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
            return Session
        except SQLAlchemyError as e:
            print(f"Erreur lors de la création de la session de base de données : {e}")
            raise e

    def get_session(self):
        """
        Retourne une nouvelle session de base de données.
        """
        try:
            session = self.Session()
            return session
        except SQLAlchemyError as e:
            print(f"Erreur lors de l'obtention de la session de base de données : {e}")
            raise e  

