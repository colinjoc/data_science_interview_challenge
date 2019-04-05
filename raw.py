import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data
head = open("header.txt","r").read().split(",\n")[:-1]
data= pd.read_csv("data.csv",names=head)

# is there any null values?
print(data.isnull().values.any())
# there is, how many are null?
print(data.isnull().sum())
# only null values in CPL_wrt_self

# look at each series
print(data["CPL_wrt_BC"0].describe())
data["CPL_wrt_BC"].hist(bins = 100)
plt.show()

# locations
print(data["client_state"].value_counts())

# duration
print(data["duration"].describe())
data["duration"].hist(bins=range(1,120))
plt.show() # spike every 6 months

# number of products
print(data["num_prods"].describe())
data["num_prods"].hist(bins = range(1,12))
print(data["num_prods"].value_counts())

# calls. Getting a weird histogram
print(data["calls"].value_counts())

# budget
print(data["avg_budget"].describe())

# business category
print(data["BC"].value_counts())

# clicks
print ( data["clicks"].describe())
data["clicks"].hist(bins = range(13,254))
plt.show()

# CPL_wrt_self. have a bunch of nans here, need to deal with them.
data["CPL_wrt_self"].dropna().hist(bins=1000)
plt.show()
# looks like a poisson distribution, have an extended tail to the right.


import scipy as sc
print(sc.stats.normaltest(data["clicks"])) # smaller than 

# want to build a model to determine the churn number. 
# will transform the location and BC into numbers using hotencoder. 
# have to deal with the nan's in CPL_wrt_self
# without nan have 8908 entries. will make a new column that signifies if the value is missing or not. Also will replace the nan values with something, maybe mean?

# add a new column that 
def transform_data(in_data):
    transformed = in_data.copy()
    transformed["CPL_wrt_self_null"] = in_data["CPL_wrt_self"].isnull()
    transformed["CPL_wrt_self"]=in_data["CPL_wrt_self"].fillna(in_data["CPL_wrt_self"].dropna().mean()) # mean

    transformed = pd.concat([transformed, pd.get_dummies(in_data["client_state"]), pd.get_dummies(in_data["BC"])] , axis = 1)
    # drop original client_state and BC columns 
    transformed=transformed.drop(["BC","client_state"],axis=1)
    target_labels= transformed["churn"]
    input_vals = transformed.loc[:,transformed.columns !="churn"]
    return input_vals,target_labels
    
# look at k-folds

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

in_vals,target=transform_data(data)



forest=RandomForestClassifier()
forest.fit(in_vals[:9000],target[:9000])

confusion_matrix(forest.predict(in_vals[9000:]), target[9000:])


cross_val_score(forest, in_vals, target, cv=3, scoring="accuracy")


