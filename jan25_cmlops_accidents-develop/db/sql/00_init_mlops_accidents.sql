--
-- Create Db, Role and User for maccidents dataset database
--
-- user dba accidents role
\connect postgres
-- create role dba
DROP ROLE IF EXISTS accidents_dba_group;
CREATE ROLE accidents_dba_group WITH
  NOLOGIN
  NOSUPERUSER
  INHERIT
  NOCREATEDB
  NOCREATEROLE
  NOREPLICATION;
-- user accidents
DROP ROLE IF EXISTS accidents_user;
CREATE ROLE accidents_user WITH
  LOGIN
  ENCRYPTED PASSWORD 'zcb8TXWa2bkX'
  SUPERUSER
  INHERIT
  CREATEDB
  CREATEROLE
  REPLICATION
  ;

--- create db if not exist
drop database IF EXISTS mlops_accidents;
create DATABASE mlops_accidents
    with
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8';

-- Grants
\connect mlops_accidents
GRANT USAGE ON SCHEMA public TO accidents_dba_group;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO accidents_dba_group;
GRANT ALL ON ALL TABLES IN SCHEMA public TO accidents_dba_group;
--
ALTER DATABASE mlops_accidents OWNER TO accidents_dba_group;
--
GRANT accidents_dba_group TO accidents_user;



