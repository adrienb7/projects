FROM apache/airflow:2.8.1

# Installation des dépendances supplémentaires pour S3 (MinIO est compatible avec les clients S3)
RUN pip install apache-airflow-providers-amazon
RUN pip install minio

# Optionnel : si tu as besoin de dépendances supplémentaires
# RUN pip install apache-airflow[aws]  

# Ajout de fichiers ou configurations supplémentaires si nécessaire
#COPY requirements.txt /requirements.txt
##RUN pip install -r /requirements.txt

# Commande pour démarrer Airflow (par défaut)
CMD ["bash", "-c", "airflow webserver & airflow scheduler"]