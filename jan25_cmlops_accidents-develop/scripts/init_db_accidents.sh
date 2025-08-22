#! /bin/bash

#
docker container stop  postgresql_accidents
docker container rm postgresql_accidents
#
docker volume rm jan25_cmlops_accidents_postgres_data_accidents

#
docker-compose -f docker-compose.yaml up -d --build postgresql_accidents