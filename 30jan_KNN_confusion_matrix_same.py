# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:59:32 2024

@author: rohit
"""

import pandas as pd
import numpy as np
wbcd=pd.read_csv("C:/2-dataset/wbcd.csv")
#there are 569 rows and 32 columns
wbcd.describe()
#in output column there is only B for Benign and
#M for Malignant
#let us first convert it as Benign and Malignant
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='B','Benign',wbcd['diagnosis']  )
#in wbcd there is column named 'diagnosis' ,where ever there
#is 'B' replace as 'Benign'
#similarity where ever there is M in the same column replace with 'Malignant'
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='M','Malignant',wbcd['diagnosis'])
################################################
#0th column is patient ID let us drop it
wbcd=wbcd.iloc[:,1:32]
################
#normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now let us apply this function to the dataframe
wbcd_n=norm_func(wbcd.iloc[:,1:32])
#because now 0 th column is output or label it is not considered hence 1:al...
###########
#
#let us now apply X as input and y as output
X=np.array(wbcd_n.iloc[:,:])
#since in wbcd n, we are already excluding output column,hence all row and..
y=np.array(wbcd['diagnosis'])
######################
#now let us split the data into training and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#here you are passing X,y instead dataframe handle
#there could chances of unbaancing of data
#let us assume you have 100 data points , out of which 80 NC and 20 cancer
#These data points must be equally distributed
#there is statified sampling concept is used
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
pd.crosstab(pred,y_test)
#let us check the applicability of the model
#i,e. miss classification , Actual patient is malignant
#i,e. cancer patient predicted but predicted is benign is 1
#actual patient is benien and predicted as cancer patient is 5
#hence this model is not acceptable
###########################################
#let us try to select correct value of k
acc=[]
#Running KNN algorithm for k=3 to 50 in the step of 2
#k value selected is odd value
for i in range(3,50,2):
    #declare the model
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc=np.mean(neigh.predict(X_train)==y_train)
    test_acc=np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
#if you will see th acc, it has got from accuracy,i[0]-train_acc
#i[1]=test_acc
#to plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
#there are 3,5,7 and 9 are possible values where accuracy is good
#let us check for k=3
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
accuracy_score(pred,y_test)
pd.crosstab(pred,y_test)
#i.e class miss classification, Actual patient is Malignant
#i,e. cancer patient but predicted is Benien is 1
#Actual patient is Benien and predicted as cancer patient is 2
#Hence this model is not acceptable
#for 5 samesinario
#for k=7 we are getting zero false positive and good accuracy
#hence k=7 is appropriate value of k 


#exploratory data analysis

#check whether data is balanced or not
#perform data analysis

def ssubset(n,m,arr1,arr2):
    set1=set(arr1)
    set2=set(arr2)
    if set2.issubset(set1):
        return 1
    else:
        return 0
    
arr1=[1,2,3,4,5]
arr2=[1,2,3]

n=len(arr1)
m=len(arr2)
result=ssubset(n,m,arr1,arr2)
print(result)


def isunion(arr1,arr2,n,m):
    union_set=set(arr1).union(arr2)
    return len(union_set)

arr1=[1,2,3,4,5,6]
arr2=[5,6,8,9]
n=len(arr1)
m=len(arr2)

result=isunion(arr1,arr2,n,m)
print(result)

