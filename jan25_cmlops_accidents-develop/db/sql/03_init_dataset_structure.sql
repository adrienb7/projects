--
-- Create schema dataset
--

-- Connect to db mlops_accidents
\connect mlops_accidents

--
-- Table 1 : te_datasets
--S
CREATE TABLE IF NOT EXISTS te_datasets (
    id INT PRIMARY KEY, -- with automatique sequence
    description VARCHAR(80) NOT NULL, 
    creation_dt TIMESTAMP NOT NULL,
    modification_dt TIMESTAMP,
    is_new BOOLEAN DEFAULT false,
    code VARCHAR(15) NOT NULL,
    type VARCHAR(10) NOT NULL
);
-- Create unique index te_dataste
CREATE UNIQUE INDEX IF NOT EXISTS ui_dataset_code ON te_datasets(code) ;
-- create index on modification_dt for te_datatset
CREATE INDEX  IF NOT EXISTS i_dataset_creation_dt ON te_datasets(modification_dt);
--
CREATE INDEX IF NOT EXISTS i_dataset_is_new ON te_datasets(is_new);
--
CREATE INDEX  IF NOT EXISTS i_dataset_type ON te_datasets(type);

-- create sequence for dataset
CREATE SEQUENCE dataset_id_seq
   START WITH 1
   INCREMENT BY 1
   NO MINVALUE
   NO MAXVALUE
   CACHE 1;


--
-- Table 2 : te_features
--
CREATE TABLE IF NOT EXISTS te_features (
    id INT PRIMARY KEY, -- with automatique sequence
    code VARCHAR(50) NOT NULL
);
-- Create unique index te_features
CREATE UNIQUE INDEX  IF NOT EXISTS ui_feature_code ON te_features(code);

-- create sequence for dataset
CREATE SEQUENCE feature_id_seq
   START WITH 1
   INCREMENT BY 1
   NO MINVALUE
   NO MAXVALUE
   CACHE 1;

--
-- Table 3 : ta_dataset_features
--
CREATE TABLE IF NOT EXISTS ta_dataset_features (
    -- id SERIAL PRIMARY KEY,
    dataset_id INT NOT NULL,
    feature_id INT NOT NULL,
    value REAL NOT NULL,
    CONSTRAINT fk_dataset FOREIGN KEY (dataset_id) REFERENCES te_datasets(id),
    CONSTRAINT fk_feature FOREIGN KEY (feature_id) REFERENCES te_features(id),
    PRIMARY KEY (dataset_id, feature_id)
);

-- Create unique index ta_dataset_features
CREATE INDEX ui_dataset_feature_id ON ta_dataset_features(dataset_id,feature_id);
