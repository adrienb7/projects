ANALYSE ET PREDICTION DE PERFORMANCE DES JOUEURS DE NBA 
------
Objectif 
-----
Conception d'un modèle de classification pour prédire la réussite ou l'échec des tirs des joueurs de NBA, en utilisant des données couvrant plus de 20 saisons (1997-2020). L'objectif était d'améliorer la précision des prédictions et d'apporter des insights sur les schémas de tirs.

Traitement des données 
-----
Travail sur un vaste ensemble de données contenant plus de 4 millions de lignes.
Application de techniques de prétraitement des données, incluant le nettoyage, la normalisation et l'intégration de données complémentaires via nba_api.
Enrichissement de l'ensemble de données par la création d'une variable personnalisée, on_fire, pour représenter l'état de forme du joueur en se basant sur ses cinq dernières performances.

Ingénierie des caractéristiques 
------

Développement de caractéristiques pertinentes pour refléter les métriques des joueurs, des matchs et des situations.
La variable on_fire a été conçue pour capturer l'élan et les séries de performances qui pourraient influencer la confiance et la performance du joueur lors des tirs.

Approche de modélisation 
-----
Implémentation de plusieurs modèles de machine learning, tels que les machines à vecteurs de support (SVM), Random Forest, XGBoost et CatBoost.
Optimisation des hyperparamètres à l'aide de GridSearchCV pour maximiser les performances du modèle et garantir une évaluation robuste.

Résultats
------
XGBoost s'est révélé être le modèle le plus performant, surpassant les autres de plus de 1 % pour certains joueurs après un réglage minutieux.
La meilleure configuration joueur/modèle atteint une accuracy avoisinant 0.7 pour la prédiction.
Une analyse comparative détaillée a montré que la combinaison de régularisation et de boosting de gradient dans XGBoost gérait efficacement les relations complexes dans les données.

Résultats et conclusions 
-------
Le projet a souligné l'importance de l'enrichissement des données et de l'ingénierie des caractéristiques dans l'analyse sportive.
La variable on_fire a apporté une puissance prédictive supplémentaire, suggérant que la forme récente est un facteur important pour prédire la réussite des tirs.
