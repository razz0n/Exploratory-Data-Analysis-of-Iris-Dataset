# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:56:01 2021

@author: RaZz oN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We load the iris dataset from the sklearn.datasets package
from sklearn.datasets import load_iris

# importing dataset to a variable
iris = load_iris()

# Type of iris which is utils.Bunch object
type(iris)

# Converting the iris dataset to dataframe with columns present
# in feature_names 
dataset = pd.DataFrame(iris.data, columns= iris.feature_names)

print(dataset)

#Adding the target column in the dataset
dataset['target'] = iris.target
print(iris.target_names)

# Now we use scatter function to visualize the dataset

# Here, iloc in used to select the 3rd column and 4th column
plt.scatter(dataset.iloc[:,2],dataset.iloc[:,3], c = iris.target)

# Now, we put labels in the plt graph as 

plt.xlabel("Petal Lenght (in cm)")
plt.ylabel(" Petal width (in cm)")
plt.legend()
plt.show()

# Now, we separate the dataset into two parts i.e 4 columns 
# and a single target column

x = dataset.iloc[:,0:4]
y = dataset.iloc[:,4]


"""
k-NN Nearest Neighbors

"""

from sklearn.neighbors import KNeighborsClassifier

# if p = 1 - manhattan if p = 2 - euclidean

# Set the model
kNN = KNeighborsClassifier(n_neighbors = 6, metric='minkowski', p=1)


# fiT the model

kNN.fit(x,y)

# Create a new sample

x_New = np.array([[5.2,3.4,1.3,0.1]])

# Predict the sample with the model

kNN.predict(x_New)

# Output is array[0] which means 0 from target i.e 
# setosa flower

"""

Using train_test_split

"""

from sklearn.model_selection import train_test_split

# Splitting the data into 20% test set and random_state provides exact same 
# data for each iterative with shuffle true and stratified 

X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.8,
                test_size=0.2, random_state = 40, shuffle= True, stratify = y)


from sklearn.neighbors import KNeighborsClassifier

#Using Manhattan distance 
kNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=1)

#Fitting the trained data
kNN.fit(X_train, y_train)

#Predicting the trained set 
predicted_y = kNN.predict(X_test)

# Now, we need to verify the predicted data with the test data 
 
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicted_y)


"""

Decision Tree 

"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Creating the model
dT = DecisionTreeClassifier()

# Fitting the model
dT.fit(X_train,y_train)

# Predicting the outcome of the model
y_predict = dT.predict(X_test)

# Calculating the accuracy of the model
accuracy_score(y_test, y_predict)

"""

Cross Validation

"""

from sklearn.model_selection import cross_val_score

# dT is the model used( here decision tree) , x =  featurecolumns , y = target
# cv = cross validation - no. of sections the test will be made (k - fold) 

score_dT = cross_val_score(dT, x , y, cv = 10)


"""

Naive Bayes

"""

from sklearn.naive_bayes import GaussianNB

nB = GaussianNB()

nB.fit(X_train,y_train)

y_pred_nB = nB.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred_nB)

from sklearn.model_selection import cross_val_score

score_nB = cross_val_score(nB,x,y, cv =10)



'''
Clustering Using K-Mean
'''
# Importing the Kmean function
from sklearn.cluster import KMeans

# Creating a model for applying K-Mean Clustering
KMNS = KMeans(n_clusters=3)

# Fitting the dataset into the model
KMNS.fit(iris_data)

# Predicting the values but this time we create a label variable to store the
# labels of the n_clusters 
Label = KMNS.predict(iris_data)

# Finding the centroid value of the clusters
cent = KMNS.cluster_centers_ 

# =============================================================================
# Plotting the datasets using scatter function
# =============================================================================

# We can only plot 2D array so we take petal values only
plt.scatter(iris_data[:,2], iris_data[:,3], c=Label, s=80)

# We plot the centroid value of each clusters
plt.scatter(cent[:,2], cent[:,3], marker='o', color='r')

# Show the plot
plt.xlabel('Petal Length(cm')
plt.ylabel('Petal width(cm')
plt.title("Data Visualization for K-Mean Clustering")
plt.show()


# =============================================================================
# Model Evaluation
# =============================================================================

# To evaluate the K-Mean Cluster, we calculate the Inertia.
KMNS.inertia_

# However, only one cluster value can't be enough to evaluate the entire
# dataset so we evaluate the dataset with k in range 0-10 for better 
# results.

# To store the inertia result for each iteration, we create an empty list.
K_inertia = []

# Applying for loop with k in range(0,10)
for i in range(1,10):
    KMNS = KMeans(n_clusters=i,random_state=30)
    KMNS.fit(iris_data)
    K_inertia.append(KMNS.inertia_)

# Plotting the result with each value of k 
plt.plot(range(1,10),K_inertia,c='green',marker='o')
plt.xlabel('No.of cluster(k)')
plt.ylabel('Inertia')
plt.title('Model Evaluation for K-Mean CLustering')
plt.show()    

'''
DBSCAN ( Density Based Spatial Clustering od Application with Noises)
'''

#Importing the DBSCAN function
from sklearn.cluster import DBSCAN

# Creating the model
dB = DBSCAN(eps = 0.6, min_samples=4)

# Fitting the model
dB.fit(iris_data)

# Labels for the data
Label = dB.labels_

# Plotting the results
plt.scatter(iris_data[:,2],iris_data[:,3],c = Label)

# Show the result 
plt.show()  

'''
Herirachical Clustering
'''
# Importing linkage, dendogram and fcluster 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Linkage to create model. Method can be single, complete or average
hR = linkage(iris_data, method='complete')

# Dendogram to plot the graph
dNd = dendrogram(hR)

# fcluster to find the labels 
Label = fcluster(hR,4, criterion='distance')


