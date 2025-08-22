import pandas as pd
import logging as log
from pathlib import Path
import sys
import os

# Global
project_dir = Path(__file__).resolve().parents[2]

# csv file name
c_2019 = 'caracteristiques-2019.csv'
c_2020 = 'caracteristiques-2020.csv'
c_2021 = 'caracteristiques-2021.csv'
c_2022 = 'caracteristiques-2022.csv'
c_2023 = 'caracteristiques-2023.csv'
#
l_2019 = 'lieux-2019.csv'
l_2020 = 'lieux-2020.csv'
l_2021 = 'lieux-2021.csv'
l_2022 = 'lieux-2022.csv'
l_2023 = 'lieux-2023.csv'
#
u_2019 = 'usagers-2019.csv'
u_2020 = 'usagers-2020.csv'
u_2021 = 'usagers-2021.csv'
u_2022 = 'usagers-2022.csv'
u_2023 = 'usagers-2020.csv'
#
v_2019 = 'vehicules-2019.csv'
v_2020 = 'vehicules-2020.csv'
v_2021 = 'vehicules-2021.csv'
v_2022 = 'vehicules-2022.csv'
v_2023 = 'vehicules-2023.csv'


#
def main():
    logger.info(f"=> Python version : {sys.version}")
    logger.info(f"=> Root Path      : {project_dir}")
    logger.info(f"=> Pandas version : {pd.__version__}")

    # load csv
    try:
        df_dic = load_csv()
    except Exception as e:
        logger.error(f"=> Load_csv error {e}")
        raise e
    
    # merge csv
    try:
        df_accidents = merge_df(df_dic)
    except Exception as e:
        logger.error(f"=> Merge df error {e}")
        raise e
    
    # missing value
    missing_values = df_accidents.isnull().sum()
    logger.info(f"=> Valeurs manquantes par colonne : {missing_values[missing_values > 0]}")

    
    # Duplicate
    logger.info(f"=> Nombre de doublons : { df_accidents.duplicated().sum()}")
    df_accidents.drop_duplicates(inplace=True)

    # Statistic sur toutes les colonnes
    logger.info(f"=> Statistic : {df_accidents.describe(include='all')}")

    # Répartition des classes de "grav"
    logger.info(f"=> Gravité : {df_accidents['grav'].value_counts()}")

    # Vma Vitesse Maximun Authorisée
    df_accidents["vma"] = pd.to_numeric(df_accidents["vma"], errors="coerce")
    logger.info(f"=> Valeurs uniques de vma AVANT filtrage : {df_accidents['vma'].unique()}")

    # Tableuw des vitesse valides
    vma_valides = [70, 90, 30, 20, 50, 60, 80, 10, 40, 110, 130, -1]

    # Filtrer uniquement les vitesses valides
    df_accidents = df_accidents[df_accidents["vma"].isin(vma_valides)]
    logger.info(f"=> Valeurs uniques de vma après filtrage : {df_accidents['vma'].unique()}")

    # Clean data
    logger.info(f"=> Normalisation des données du dataframe df_accidents {df_accidents.info}")
    try:
        clean_data(df_accidents)
    except Exception as e:
        logger.error(f"=> Clean du dataframe accidents error : {e}")
        raise e    
        

    #Sauvegarde
    try:, dtype=str, sep=";"
        save_df(df_accidents, 'df_accidents')
    except Exception as e:
        logger.error(f"=> Sauvergarde du dataframe accidents error : {e}")
        raise e    


#
def load_csv():
    logger.info("=> load csv")
    # dictionnaire des df
    df_dic = {}
    try:
        df_dic['df_c_2019'] = pd.read_csv(f"{project_dir}/data/raw/{c_2019}", dtype=str, sep=";")
        df_dic['df_c_2020'] = pd.read_csv(f"{project_dir}/data/raw/{c_2020}", dtype=str, sep=";")
        df_dic['df_c_2021'] = pd.read_csv(f"{project_dir}/data/raw/{c_2021}", dtype=str, sep=";")
        df_dic['df_c_2022'] = pd.read_csv(f"{project_dir}/data/raw/{c_2022}", dtype=str, sep=";")
        df_dic['df_c_2023'] = pd.read_csv(f"{project_dir}/data/raw/{c_2023}", dtype=str, sep=";")
        #
        df_dic['df_l_2019'] = pd.read_csv(f"{project_dir}/data/raw/{l_2019}", dtype=str, sep=";")
        df_dic['df_l_2020'] = pd.read_csv(f"{project_dir}/data/raw/{l_2020}", dtype=str, sep=";")
        df_dic['df_l_2021'] = pd.read_csv(f"{project_dir}/data/raw/{l_2021}", dtype=str, sep=";")
        df_dic['df_l_2022'] = pd.read_csv(f"{project_dir}/data/raw/{l_2022}", dtype=str, sep=";")
        df_dic['df_l_2023'] = pd.read_csv(f"{project_dir}/data/raw/{l_2023}", dtype=str, sep=";")
        #   
        df_dic['df_u_2019'] = pd.read_csv(f"{project_dir}/data/raw/{u_2019}", dtype=str, sep=";")   
        df_dic['df_u_2020'] = pd.read_csv(f"{project_dir}/data/raw/{u_2020}", dtype=str, sep=";")   
        df_dic['df_u_2021'] = pd.read_csv(f"{project_dir}/data/raw/{u_2021}", dtype=str, sep=";")   
        df_dic['df_u_2022'] = pd.read_csv(f"{project_dir}/data/raw/{u_2022}", dtype=str, sep=";")   
        df_dic['df_u_2023'] = pd.read_csv(f"{project_dir}/data/raw/{u_2023}", dtype=str, sep=";")
        #   
        df_dic['df_v_2019'] = pd.read_csv(f"{project_dir}/data/raw/{v_2019}")  
        df_dic['df_v_2020'] = pd.read_csv(f"{project_dir}/data/raw/{v_2020}", dtype=str, sep=";")  
        df_dic['df_v_2021'] = pd.read_csv(f"{project_dir}/data/raw/{v_2021}", dtype=str, sep=";")  
        df_dic['df_v_2022'] = pd.read_csv(f"{project_dir}/data/raw/{v_2022}", dtype=str, sep=";")  
        df_dic['df_v_2023'] = pd.read_csv(f"{project_dir}/data/raw/{v_2023}", dtype=str, sep=";")
        
    except Exception as e:
        raise e
    return df_dic

def merge_df(df_dic):
    logger.info(f"=> merge df")
    try:
        # Merge des datatsets de 2019 sur "Num_Acc"
        df_2019 = df_dic['df_c_2019'].merge(df_dic['df_l_2019'], on="Num_Acc")\
                                  .merge(df_dic['df_v_2019'], on="Num_Acc")\
                                  .merge(df_dic['df_u_2019'], on=["Num_Acc", "id_vehicule"])
        logger.info(f"=> df_2019 : {df_2019.tail()}")

        #  Fusion des fichiers de 2020 sur "Num_Acc"
        df_2020 = df_dic['df_c_2020'].merge(df_dic['df_l_2020'], on="Num_Acc")\
                                  .merge(df_dic['df_v_2020'], on="Num_Acc")\
                                  .merge(df_dic['df_u_2020'], on=["Num_Acc", "id_vehicule"])
        logger.info(f"=> df_2020 : {df_2020.tail()}")

        # Fusion des fichiers de 2021
        df_2021 = df_dic['df_c_2021'].merge(df_dic['df_l_2021'], on="Num_Acc")\
                                  .merge(df_dic['df_v_2021'], on="Num_Acc")\
                                  .merge(df_dic['df_u_2021'], on=["Num_Acc", "id_vehicule"])
        logger.info(f"=> df_2021 : {df_2021.tail()}")
        
        #  Fusion des fichiers de 2022 (rename colonnes Accident_Id by "Num_Acc)
        df_dic['df_c_2022'].rename(columns={"Accident_Id": "Num_Acc"}, inplace=True)
        df_2022 = df_dic['df_c_2022'].merge(df_dic['df_l_2022'], on="Num_Acc")\
                                  .merge(df_dic['df_v_2022'], on="Num_Acc")\
                                  .merge(df_dic['df_u_2022'], on=["Num_Acc", "id_vehicule"])
        logger.info(f"=> 2022 : {df_2022.tail()}")

        #  Fusion des fichiers de 2023
        df_2023 = df_dic['df_c_2023'].merge(df_dic['df_l_2023'], on="Num_Acc")\
                                  .merge(df_dic['df_v_2023'], on="Num_Acc")\
                                  .merge(df_dic['df_u_2023'], on=["Num_Acc", "id_vehicule"])
        logger.info(f"=> 202 : {df_2023.tail()}")
        
        ## Concaténation de toutes les années
        df_accidents = pd.concat([df_2019, df_2020, df_2021, df_2022, df_2023], axis=0, ignore_index=True)
        logger.info(f"=> df accidents      : {df_accidents.head()}")
        logger.info(f"=> df accidents info : {df_accidents.info()}")

        #
        return df_accidents
    except Exception as e:
        raise e


# Clean df_accidents
def clean_data(data):
    # 1. Remplacer les virgules par des points dans les colonnes décimales
    data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    
    # 2. Convertir toutes les colonnes numériques au bon format
    # Nous utilisons `errors='ignore'` pour que les colonnes non numériques ne soient pas affectées
    data = data.apply(pd.to_numeric, errors='ignore')
    
    # 3. Supprimer les colonnes spécifiques
    columns_to_drop = ['Num_Acc', 'v1', 'v2', 'Acc', 'dep', 'com', 'int', 'adr', 'voie', 'lartpc', 
                       'larrout', 'id_vehicule', 'trajet', 'pr', 'pr1', 'num_veh_x', 'occutc', 'num_veh_y', 
                       'actp', 'etatp', 'id_usager']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # 4. Décrémenter les valeurs de la colonne 'grav' de 1
    if 'grav' in data.columns:
        data['grav'] = data['grav'] - 1
    
    # 5. Nettoyage de la colonne 'nbv' : supprimer les valeurs inférieures à 0 et les valeurs non numériques
    if 'nbv' in data.columns:
        # Convertir 'nbv' en numérique (les erreurs seront transformées en NaN)
        data['nbv'] = pd.to_numeric(data['nbv'], errors='coerce')
        
        # Supprimer les lignes où 'nbv' est inférieure à 0 ou NaN
        data = data[data['nbv'] >= 0]
        data = data.dropna(subset=['nbv'])
        
        # Convertir 'nbv' en entier (si nécessaire)
        data['nbv'] = data['nbv'].astype(int)
    
    # Afficher les premières lignes pour vérifier le résultat
    print(data.head())
    
    # Sauvegarder les données nettoyées dans un nouveau fichier CSV
    #output_filename = "df_accidents_nettoye.csv"  # Nouveau fichier après nettoyage
    #data.to_csv(output_filename, sep=";", index=False)


def save_df(df, filename):
    logger.info("=> Sauvegarde df")
    try:
        output_filepath = os.path.join(project_dir, f'data/preprocessed/{filename}.csv')
        df.to_csv(output_filepath, index=False, encoding="utf-8")
    except Exception as e:
        raise e

if __name__ == '__main__':
    #
    log.basicConfig(filename=f"{project_dir}/accidents.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=log.DEBUG)
    logger = log.getLogger('make_dataset')
    #
    main()
