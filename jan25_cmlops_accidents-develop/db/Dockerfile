FROM postgres:16.3-bullseye
VOLUME /var/lib/postgresql/backup
#VOLUME /var/lib/postgresql/python
#COPY python/*.py /var/lib/postgresql/python/
#COPY conf/pg_hba.conf /var/lib/postgresql/data/
#Install python3.7
#RUN cat /etc/apt/sources.list
#RUN apt-get update && apt-get -y upgrade
#RUN apt-get -y install python3-pip && apt-get -y install postgresql-plpython3-16
#COPY python/requirements.txt .
#RUN pip3 install -r ./requirements.txt
#posgresql
COPY sql/*.sql /docker-entrypoint-initdb.d/
RUN chmod +rx /docker-entrypoint-initdb.d/*.sql
CMD ["docker-entrypoint.sh", "postgres"]