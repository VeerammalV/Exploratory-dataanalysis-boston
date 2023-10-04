#matplotlib inline
import numpy as np
import pandas as pd
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
boston_dst = load_boston() 
#import seaborn as sns
#iris = sns.load_dataset('iris')

print("Type of the dataset: ", type(boston_dst))
print("Keys: ", boston_dst.keys())
print(boston_dst.filename)
print(boston_dst.DESCR)
print("Type  of data: ", type(boston_dst.data))
print("Shape of data: ", boston_dst.data.shape)
print("Feature Names: ", boston_dst.feature_names)
print("Type  of target: ", type(boston_dst.target))
print("Shape of target: ", boston_dst.target.shape)
boston_pd = pd.DataFrame(boston_dst.data)
boston_pd.head()
boston_pd.columns = boston_dst.feature_names
boston_pd.head()
boston_pd['PRICE'] = boston_dst.target
boston_pd.head()
#Exploratory Data Analysis
boston_pd #Overview of the Data
boston_pd.describe(include='all').transpose() #understand characteristics of the data and to get a quick summary of it.
boston_pd.PRICE.value_counts() #get count of each category in a categorical attributed series of values.
print(boston_pd.nunique()) #number of unique elements in each column.
boston_pd[boston_pd.isnull().any(axis=1)] #Display Rows with Missing Data
boston_pd.isnull().sum() #Count the number of missing values for each column

total = boston_pd.isnull().sum().sort_values(ascending=False)
percent = (boston_pd.isnull().sum()/boston_pd.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, 
                         keys=['Total', 'Percent'])
missing_data.head(20)

boston_pd.notna().sum() #count the number of non-missing data
boston_pd = boston_pd.dropna() #drop null values 
boston_pd.shape
boston_pd = boston_pd.drop((missing_data[missing_data['Total'] > 1]).index, 1)
boston_pd = boston_pd.drop(boston_pd.loc[boston_pd['CRIM'].isnull()].index)
boston_pd.isnull().sum().max()
boston_pd.AGE = boston_pd.AGE.fillna(35)
boston_pd.fillna(boston_pd.mean())

#BoxPlot
plt.boxplot(list(boston_pd.PRICE)); 
plt.show(); 
fig,  ax = plt.subplots(len(list(boston_pd.columns)), figsize=(8,40))
#one type
for i, feature_name in enumerate(list(boston_pd.columns)):
    sns.boxplot(y=boston_pd[feature_name], ax=ax[i]);
    ax[i].set_xlabel(feature_name, fontsize=8);
    #ax[i].set_title("Box plot {} ".format(feature_name), fontsize=8);

plt.show();
#Scatterplot
for feature_name in boston_dst.feature_names:
    plt.figure(figsize=(5, 4));
    plt.scatter(boston_pd[feature_name], boston_pd['PRICE']);
    plt.ylabel('Price', size=12);
    plt.xlabel(feature_name, size=12);

plt.show();
#implot
sns.lmplot(x = 'RM', y = 'PRICE', data = boston_pd)

#Histogram
plt.figure(figsize=(8, 6));
plt.hist(boston_pd['PRICE']);
plt.title('Boston Housing Prices and Count Histogram');
plt.xlabel('price ($1000s)');
plt.ylabel('count');
plt.show();
from scipy.stats import norm
sns.distplot(boston_pd['PRICE'], fit=stats.norm);
plt.figure(figsize=(8, 6));
res = stats.probplot(boston_pd['PRICE'], plot=plt);

fig,  ax = plt.subplots(len(list(boston_pd.columns)), figsize=(12,46))
#another one type
for i, feature_name in enumerate(list(boston_pd.columns)):
    if (feature_name != 'CHAS'):
       sns.distplot(boston_pd[feature_name], hist=True, ax=ax[i]);
       ax[i].set_ylabel('Count', fontsize=8);
       ax[i].set_xlabel(" {}".format(feature_name), fontsize=8);
       #ax[i].set_title("Freq dist "+feature_name, fontsize=8);

plt.show();
#pairplot

sns.pairplot(boston_pd); #easy one

#another one type
n = 4
for i in range(0, len(boston_pd.columns), n):
    sns.pairplot(data=boston_pd,
                x_vars=boston_pd.columns[i:i+n],
                y_vars=['PRICE']);

#Heatmap: Two-Dimensional Graphical Representation              
plt.figure(figsize=(12, 9));
correlation_matrix = boston_pd.corr().round(2);
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True);
#heatmap
sns.heatmap(correlation_matrix[(correlation_matrix >= 0.5) | (correlation_matrix <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
