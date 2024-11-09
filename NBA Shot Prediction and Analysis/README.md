# NBA PROJECT

Objective: Designed a classification model to predict whether an NBA player’s shot would be successful or not, leveraging data from over 20 seasons (1997-2020). The goal was to enhance predictive accuracy and offer insights into shooting patterns.

Data Handling:

Worked with a large-scale dataset consisting of over 4 million records.
Applied data preprocessing techniques, including cleaning, normalization, and integration of additional data from the nba_api.
Enriched the dataset by creating a custom variable, on_fire, to represent the player's current form based on their past five game performances.
Feature Engineering:

Developed relevant features to reflect player, game, and situational metrics.
The on_fire feature was engineered to capture momentum and streaks that may influence a player’s shooting confidence and performance.
Modeling Approach:

Implemented various machine learning models, including Support Vector Machines (SVM), Random Forest, XGBoost, and CatBoost.
Conducted hyperparameter tuning using GridSearchCV to optimize model performance and ensure robust evaluation.
Results:

XGBoost emerged as the leading model, outperforming others by over 1% for certain players after extensive tuning.
Detailed comparative analysis demonstrated that XGBoost's combination of regularization and gradient boosting effectively handled complex relationships in the data.
Outcome and Insights:

The project underscored the importance of data enrichment and feature engineering in sports analytics.
The on_fire variable provided additional predictive power, suggesting that recent form is a significant factor in shot success prediction.

