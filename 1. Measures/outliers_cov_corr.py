import os
os.chdir('C:/Training/Academy/Statistics (Python)/Datasets')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
iris = pd.read_csv("iris.csv")

iris["Petal.Width"].kurtosis() 

plt.hist(iris["Petal.Width"])

diamonds = pd.read_csv("Diamonds.csv")

diamonds['price'].kurtosis()

plt.hist(diamonds['price'])

####################### Covariance ########################

a = np.array([2,4,5,7,9,13,20])
b = np.array([45,43,36,28,23,19,10])

np.cov(a,b)

np.corrcoef(a,b)
plt.scatter(a, b)
plt.show()

cts = diamonds.groupby('cut')['price'].mean()
cts1 = cts.reset_index()
sns.barplot(data=cts1, x='cut', y='price')
plt.ylabel("Mean Price")
plt.show()


diamonds.corr()

sns.heatmap(
    diamonds.corr(),
    xticklabels=diamonds.corr().columns, 
    yticklabels=diamonds.corr().columns, 
    annot=True)
plt.show()

## Generate heatmaps
## Boston
## Iris
## Cars93
## Exp_Salaries
## milk

sns.pairplot(iris)
plt.show()

#################### Outliers ###########

plt.boxplot(diamonds['price'])
plt.show()

q1 = diamonds['price'].quantile(q=0.25)
q3 = diamonds['price'].quantile(q=0.75)
iqr = q3 - q1
print("Inter-Quartile Range =",iqr)

lim_iqr = 1.5*iqr

upper_iqr = q3 + lim_iqr
lower_iqr = q1 - lim_iqr

outlier_df = diamonds[(diamonds['price'] > upper_iqr) | (diamonds['price'] < lower_iqr)]
outlier_df['price']



def detect_outliers(df, column):
    q1 = df[column].quantile(q=0.25)
    q3 = df[column].quantile(q=0.75)
    iqr = q3 - q1

    lim_iqr = 1.5*iqr

    upper_iqr = q3 + lim_iqr
    lower_iqr = q1 - lim_iqr

    outliers_df = df[(df[column] > upper_iqr) | (df[column] < lower_iqr)]  
    return outliers_df[column].to_list();

detect_outliers(diamonds,'price')


sns.boxplot(y='price', data=diamonds)
sns.swarmplot(y=detect_outliers(diamonds, 'price'), data=diamonds)
plt.show()


housing = pd.read_csv("Housing.csv")

sns.boxplot(y='price', data=housing)
sns.swarmplot(y=detect_outliers(housing, 'price'), data=housing)
plt.show()
