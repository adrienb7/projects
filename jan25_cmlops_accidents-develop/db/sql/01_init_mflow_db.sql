-- user mflow role
\connect postgres
-- create role dba
DROP ROLE IF EXISTS mlflow_dba_group;
CREATE ROLE mlflow_dba_group WITH
  NOLOGIN
  NOSUPERUSER
  INHERIT
  NOCREATEDB
  NOCREATEROLE
  NOREPLICATION;
-- user amlflow
DROP ROLE IF EXISTS mlflow_user;
CREATE ROLE mlflow_user WITH
  LOGIN
  ENCRYPTED PASSWORD 'zcb8TXWa2bkY'
  SUPERUSER
  INHERIT
  CREATEDB
  CREATEROLE
  REPLICATION
  ;

--- create db if not exist
drop database IF EXISTS mlflow;
create DATABASE mlflow
    with
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8';

-- Grants
\connect mlflow
GRANT USAGE ON SCHEMA public TO mlflow_dba_group;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO mlflow_dba_group;
GRANT ALL ON ALL TABLES IN SCHEMA public TO mlflow_dba_group;
--
ALTER DATABASE mlflow OWNER TO mlflow_dba_group;
--
GRANT mlflow_dba_group TO mlflow_user;


