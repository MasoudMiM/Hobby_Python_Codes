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


