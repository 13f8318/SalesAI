import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from xgboost import XGBClassifier
import matplotlib.path as mpath
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
import os
import csv
from os import listdir
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import copy
import operator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


df_train = pd.read_csv('/home/hassan/Downloads/data_format1/train_format1.csv', skipinitialspace=True, usecols=['user_id','merchant_id', 'label'])
df_train = df_train.rename(columns={'merchant_id': 'seller_id'})
#df_user_related = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/user-related-features.csv', skipinitialspace=True)
df_interactive = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/interactive-features.csv', skipinitialspace=True)
del df_interactive['life_span']

#This user related is without ratio features.
df_user_related = pd.read_csv('/home/hassan/Desktop/My Current Projects/SalesAI/Algorithm training files/user-related-features.csv', skipinitialspace=True)
#df_train = df_train.merge(df_user_related, how="left", on=['user_id'])
df_interactive = df_interactive.merge(df_user_related, how="left", on=['user_id'])

df_train = df_train.merge(df_interactive, how="left", on=['user_id', 'seller_id'])

y = pd.DataFrame()
y['label'] = df_train['label']

del df_train['label']

print('Splitting.')
X_train, X_test, y_train, y_test = train_test_split(df_train, y.values.ravel(), test_size=0.25, random_state=42)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
# y_train= y_train.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)


#X_train.to_csv('/home/hassan/Desktop/My Current Projects/SalesAI/Algorithm training files/training_interactive.csv', index=False)
#Merge User-related-features and train_format1.csv

# print
print('Length of Training set: ', len(X_train))
print('Length of Test set: ', len(X_test))

print(y_test[0])

#model = XGBClassifier(learning_rate=1.5, n_estimators=150, max_depth=60)
model = AdaBoostClassifier(n_estimators=150)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
correct =0
totalones=0
totalonesp= 0
totalzerosp= 0
totalzeros=0
for  i in range(0, len(y_test)):
    if y_test[i] == predictions [i]:
        correct+=1
    if y_test[i] == 1:
        totalones+=1
    if y_test[i] == 0:
        totalzeros+=1
    if y_test[i] == predictions[i] == 1:
        totalonesp+=1
    if y_test[i] == predictions[i] == 0:
        totalzerosp+=1


print('Total length of ytest: ', len(y_test))
print('Total Correctly Predicted: ', correct)
print('Total no of ones: ', totalones)
print('Total ones correctly predicted: ', totalonesp)
print('Total Zeros: ', totalzeros)
print('Total zeros correctly predicted: ', totalzerosp)
print('Correct Percentage: ', (correct/len(y_test)*100))

print('------------------Second XgBoost-------------------------')
clf = GradientBoostingClassifier(n_estimators=150, learning_rate=1.5,
     max_depth=1, random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))



