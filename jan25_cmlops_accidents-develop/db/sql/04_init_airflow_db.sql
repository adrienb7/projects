-- user mflow role
\connect postgres
-- create role dba
DROP ROLE IF EXISTS airflow_dba_group;
CREATE ROLE airflow_dba_group WITH
  NOLOGIN
  NOSUPERUSER
  INHERIT
  NOCREATEDB
  NOCREATEROLE
  NOREPLICATION;
-- user airflow
DROP ROLE IF EXISTS airflow_user;
CREATE ROLE airflow_user WITH
  LOGIN
  ENCRYPTED PASSWORD 'zcb8TXWa2bkK'
  SUPERUSER
  INHERIT
  CREATEDB
  CREATEROLE
  REPLICATION
  ;

--- create db if not exist
drop database IF EXISTS airflow;
create DATABASE airflow
    with
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8';

-- Grants
\connect airflow
GRANT USAGE ON SCHEMA public TO airflow_dba_group;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO airflow_dba_group;
GRANT ALL ON ALL TABLES IN SCHEMA public TO airflow_dba_group;
--
ALTER DATABASE airflow OWNER TO airflow_dba_group;
--
GRANT airflow_dba_group TO airflow_user;


