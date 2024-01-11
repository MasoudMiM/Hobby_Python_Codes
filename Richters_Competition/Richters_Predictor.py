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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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

############### Let's define a function that takes in a model and returns the f1 score
X_train, X_val, y_train, y_val = train_test_split(df_train_values, df_train_labels['damage_grade'], train_size=0.8)
# the function will include the data normalization and the grid search
# the function will also print the best parameters and the best score
# the function will also print the f1 score for the validation set
# the function should also return the model so it can be applied to the test data later on
def model_f1_score(model, X_train, y_train, X_val, y_val):
    # data normalization
    min_max_scaler = MinMaxScaler()
    x_train_scaled = min_max_scaler.fit_transform(X_train)
    x_val_scaled = min_max_scaler.fit_transform(X_val)
    # defining the parameter range for five different models, including 
    # logistic regression, random forest, gradient boosting, support vector machine, and k-nearest neighbors
    n_sampled_parameters = 5
    if model == logreg:
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  
              'penalty': ['l1', 'l2']}
        grid = RandomizedSearchCV(model, param_grid, n_iter=5, refit = True, n_jobs=n_sampled_parameters, verbose = 1)
    elif model == rf:
        param_grid = {'n_estimators': [100, 200, 300, 400, 500],  
              'max_depth': [5, 10, 15, 20, 25]}
        grid = RandomizedSearchCV(model, param_grid, n_iter=5, refit = True, n_jobs=n_sampled_parameters, verbose = 1)
    elif model == gb:
        param_grid = {'n_estimators': [100, 200, 300, 400, 500],  
              'max_depth': [5, 10, 15, 20, 25]}
        grid = RandomizedSearchCV(model, param_grid,  n_iter=5, refit = True, n_jobs=n_sampled_parameters, verbose = 1)
    elif model == svc:
        param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = RandomizedSearchCV(model, param_grid, n_iter=5, refit = True, n_jobs=n_sampled_parameters, verbose = 1)
    elif model == knn:
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11],  
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
        grid = RandomizedSearchCV(model, param_grid, n_iter=5, refit = True, n_jobs=n_sampled_parameters, verbose = 1)
    else:
        print("The model is not defined!")

    # developing the model using the grid search best parameters
    grid.fit(x_train_scaled, y_train)
    print(f"The best parameters are {grid.best_params_}")
    print(f"The best score is {grid.best_score_:.3f}")
    # predict the validation set
    y_pred = grid.predict(x_val_scaled)
    print(f"The f1 score for the logistic regression model is {f1_score(y_val, y_pred, average='micro'):.3f}")
    return grid

# let's define try five different models and see which one performs the best
# includeing logistic regression, random forest, gradient boosting, support vector machine, and k-nearest neighbors

logreg = LogisticRegression(max_iter=500)
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svc = SVC()
knn = KNeighborsClassifier()

model_dic = {}
models = [logreg, rf, gb, svc, knn]
model_names = ['logreg', 'rf', 'gb', 'svc', 'knn']
for model, model_name in zip(models, model_names):
    print(f"Model: {model_name}")
    model_dic[model_name]=model_f1_score(model, X_train, y_train, X_val, y_val)
    print("\n")

# let's try to use the best model to predict the test data
model_selected = input("Please select the model you want to use for prediction: ")
df_test_values = pd.read_csv(f"{directory}test_values.csv")
# converting the categorical variables into numerical values.
for column in df_test_values.select_dtypes(include=['object']):
    df_test_values[column],_ = pd.factorize(df_test_values[column])
    # data normalization
    min_max_scaler = MinMaxScaler()
    x_train_scaled = min_max_scaler.fit_transform(df_train_values)
    x_test_scaled = min_max_scaler.fit_transform(df_test_values)
    # predict the test set
    y_pred = model_dic[model_selected].predict(x_test_scaled)

# let's now train a neural network model and see how it performs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# let's define the model
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)
])

nn_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# ler's normalize the data and then train the model
min_max_scaler = MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(X_train)
x_val_scaled = min_max_scaler.fit_transform(X_val)
nn_model.fit(x_train_scaled, y_train, epochs=10)


# let's try to use the model to predict the test data
df_test_values = pd.read_csv(f"{directory}test_values.csv")
# converting the categorical variables into numerical values.
for column in df_test_values.select_dtypes(include=['object']):
    df_test_values[column],_ = pd.factorize(df_test_values[column])
    # data normalization
    min_max_scaler = MinMaxScaler()
    x_train_scaled = min_max_scaler.fit_transform(df_train_values)
    x_test_scaled = min_max_scaler.fit_transform(df_test_values)
    # predict the test set
    y_pred = nn_model.predict(x_test_scaled)
    y_pred = np.argmax(y_pred, axis=1)

# put the results into a csv file
df_test_values['damage_grade'] = y_pred
df_test_values[['building_id', 'damage_grade']].to_csv(f"{directory}submission.csv", index=False)


