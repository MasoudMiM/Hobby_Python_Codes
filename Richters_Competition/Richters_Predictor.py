# This is a code written to pariticpate in the Richters Predictor: Modeling Earthquake Damage competition on 
# DrivenData (https://www.drivendata.org/competitions/57/nepal-earthquake/).

# based on the blog posted by the competition: (see https://drivendata.co/blog/richters-predictor-benchmark)
# goal is to predict the level of damage a building suffered as a result of the 2015 earthquake. 
# The data comes from the 2015 Nepal Earthquake Open Data Portal, and mainly consists of information on the buildings' 
# structure and their legal ownership. Each row in the dataset represents a specific 
# building in the region that was hit by Gorkha earthquake.


directory = "Richters_Competition/"
############# Importing the libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

############# Looking into the features, initial exploratory data analysis, and some initial preparation
df_train_values = pd.read_csv(f"{directory}train_values.csv")
df_train_labels = pd.read_csv(f"{directory}train_labels.csv")
print(f"The size of the feature dataframe is{df_train_values.shape}")
print(f"The number of missing values in the dataset {df_train_values.isna().sum().sum()}" )

# converting the categorical variables into numerical values.
for column in df_train_values.select_dtypes(include=['object']):
    df_train_values[column],_ = pd.factorize(df_train_values[column])


# a quick look at the correlation of the features with the target variable "damage_grade"
df_train_total = pd.merge(df_train_values, df_train_labels, on='building_id')
corr_mat = df_train_total.corr()
upper_corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)) 
unique_corr_pairs = upper_corr_mat.unstack().dropna() 

print("Sort correlation matrix without duplicates:") 
sorted_mat = unique_corr_pairs.sort_values() 
for item, value in sorted_mat.items():
    print(np.round(value,4), item)
# put the results to a csv file
sorted_mat.to_csv(f"{directory}correlation_values.csv", index_label=['variable1', 'variable2', 'corr_coef'])

plt.figure(figsize=(30,30), dpi =150)
sns.heatmap(corr_mat,annot=True)
plt.savefig(f"{directory}correlation_matrix.png")

# a quick look at the pairplot for the variables (This can take a long time...)
#plt.figure(figsize=(30,30), dpi=150)
#sns.pairplot(df_train_total, hue="damage_grade")
#plt.savefig("Richters_Competition/pairplot.png")



############# Initial attempt - creating a logistic regression model using all features
X_train, X_val, y_train, y_val = train_test_split(df_train_values, df_train_labels['damage_grade'], train_size=0.8)

# data normalization
min_max_scaler = MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(X_train)
x_val_scaled = min_max_scaler.fit_transform(X_val)

# developing the logistic regression model - no hyperparameter tuning
logreg = LogisticRegression()
logreg.fit(x_train_scaled, y_train) 
y_pred = logreg.predict(x_val_scaled)
print(f"The f1 score for the logistic regression model is {f1_score(y_val, y_pred, average='micro'):.3f}")

# Now, let's do a grid search and see if we can improve the model
# defining the parameter range
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  
              'penalty': ['l1', 'l2']}
grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 1)
# developing the model using the grid search best parameters
grid.fit(x_train_scaled, y_train)
print(f"The best parameters are {grid.best_params_}")
print(f"The best score is {grid.best_score_:.3f}")
y_pred = grid.predict(x_val_scaled)
print(f"The f1 score for the logistic regression model is {f1_score(y_val, y_pred, average='micro'):.3f}")
# NOTE: Even with the grid search, the f1 score is not improved.

