# PREDIR LE COURS DE CRYPTOMONNAIES EST-IL ENVISEAGEABLE AVEC DES OUTILS DE ML ?

## Objectif
Conception d'un modèle de prévision des séries temporelles pour prédire les variations des prix du Bitcoin, en utilisant des données historiques couvrant plusieurs années. L'objectif était d'améliorer la précision des prévisions et d'explorer l'efficacité de divers modèles de machine learning sur des données volatiles.

## Traitement des données
Travail sur un vaste ensemble de données comprenant les prix horaires du Bitcoin sur plusieurs années. Application de techniques de prétraitement des données incluant le nettoyage, la gestion des valeurs manquantes et la transformation des données en séries temporelles hebdomadaires et mensuelles pour capturer la saisonnalité.

## Ingénierie des caractéristiques
Création de variables pour refléter les tendances des prix et les effets de saisonnalité. Intégration de variables de délai (lag) et de moyennes mobiles pour représenter la dynamique des séries temporelles. Les caractéristiques ont été choisies pour capturer les fluctuations à court et long terme des prix.

## Approche de modélisation
Implémentation de plusieurs modèles de machine learning, incluant LSTM, XGBoost, Prophet et ARIMA. Optimisation des hyperparamètres via GridSearchCV et autres techniques pour maximiser les performances des modèles. Chaque modèle a été évalué en fonction de sa capacité à prédire les valeurs futures des séries temporelles.

## Résultats
Le modèle Prophet a montré des performances prometteuses sur des séries temporelles présentant des effets saisonniers marqués, tandis que XGBoost a su gérer efficacement la complexité des données, surpassant ARIMA et LSTM dans certaines configurations. Le meilleur modèle a atteint un MSE de 3654149,48 et un MAE de 1678,49, mettant en lumière la volatilité intrinsèque des prix du Bitcoin.

## Résultats et conclusions
Le projet a démontré l'importance de l'intégration de modèles robustes pour gérer la volatilité et les influences externes du marché des cryptomonnaies. La combinaison de modèles traditionnels et avancés a permis d'améliorer la précision des prévisions. Pour des prévisions plus précises, l'intégration de données externes telles que les actualités et les tendances des réseaux sociaux pourrait apporter un bénéfice supplémentaire.
Nous pourrons essayer d'intégrer le modèle GARCH pour capturer et prédire la volatilité des prix des cryptomonnaies et ainsi améliorer la précision des prévisions.






