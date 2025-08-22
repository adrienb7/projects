FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier fichiers
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

#COPY app/ /app/

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

#FROM python:3.8-alpine instable pour mlflow
#
## Env vars
#ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONUNBUFFERED 1
#
## Install build dependencies
#RUN apk update && apk add --no-cache \
#    gcc \
#    g++ \
#    libffi-dev \
#    musl-dev \
#    openblas-dev \
#    lapack-dev \
#    python3-dev \
#    py3-pip \
#    linux-headers \
#    build-base \
#    && rm -rf /var/cache/apk/*
#WORKDIR /app
#
## Copy files
#COPY app requirements.txt ./
#
## Install Python dependencies
#RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
#
## Copier tout le reste du projet
#COPY app/ /app/
#
#EXPOSE 80
#
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
#