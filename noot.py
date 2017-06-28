# Based on https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
#%matplotlib inline

print("Loading dataset.")
data = pd.read_csv('data.csv')

print("Filling zeroes with means.")
data_zero_corrected = data.replace(0,data.mean())

print("Filling empty cells by columnwise average.")
data_filled = data_zero_corrected.fillna(data_zero_corrected.mean())
#for colname, series in data.iteritems():
#	series.fillna(series.mean, inplace=True)
#data.apply(lambda x: x.fillna(x.mean(), inplace=True),axis=0)
print("Corrected series:\n")
print(str(data_filled))

print("Converting to np.array.")
X=data_filled.values

print("Scaling the values.\n")
X = scale(X)

print("Computing PCA.\n")
pca = PCA(n_components=10)
print(str(pca))

print("Fitting PCA to scaled values.\n")
pca.fit(X)
print(str(X))

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
#var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(str(var))

# Write results out
pd.DataFrame(pca.components_).to_csv('principal_components.csv')
pd.DataFrame(var).to_csv('explained_variance_ratios.csv')

# Snip out all non-rating columns - no age, sex, or mental illness.
X_sliced = scale(X[5:,:])
print("Second PCA ignoring population statistics.  Raw values:")
print(str(X_sliced))

pca_sliced = PCA(n_components=10)
print(str(pca_sliced))
pca_sliced.fit(X_sliced)
print(str(X_sliced))
pd.DataFrame(pca_sliced.components_).to_csv('sliced_principal_components.csv')
pd.DataFrame(pca.explained_variance_ratio_).to_csv('sliced_explained_variance_ratios.csv')