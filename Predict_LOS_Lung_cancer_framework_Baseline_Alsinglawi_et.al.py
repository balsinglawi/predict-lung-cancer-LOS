# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:59:39 2021

@author: Belal Alsinglawi
""""

# import essential Libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import plotly.express as px
plt.rc("font", size=14)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks    
from imblearn.metrics import sensitivity_specificity_support
from imblearn.metrics import classification_report_imbalanced

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, average_precision_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from time import time
import shap

data = pd.read_csv('MIMIC.csv', header = 0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
# descriptive Stats
data_Stats = data.describe()

# PLOT LOS without sampling
plt.title('MIMIC-III Lung Cancer distinct admissions\n LOS Distribution Without Class Balancing', fontsize=14) # without sampling
sns.countplot(data['LOS_x_Short_Long'], palette='colorblind') #https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
plt.ylabel('Patients Count Per Class')
plt.xlabel('Short LOS: 7< days Class "0" \n Long LOS: 7> days Class "1" ')
plt.show()
plt.savefig('LOS_x_Short_Long_distribution')

X = data.loc[:, data.columns != 'LOS_x_Short_Long']
y = data.loc[:, data.columns == 'LOS_x_Short_Long']

"""
#RFE: Recursive Feature Elimination specific to each model choice

# Create the RFE object and rank clinical features - LR
logreg = LogisticRegression()
rfe = RFE(logreg,60)
rfe_features = rfe.fit(X, y)
# Plot pixel ranking
print(rfe_features.support_)
print(rfe_features.ranking_)
rfe_ranking = rfe_features.ranking_
"""
#RFE: for each model: RF, LR etc
cols = data[rfe_ranking]
X = cols  # RFE
#___Baselining with Cross-Validation

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
model = RandomForestClassifier(random_state=42)
# evaluate model
start = time()
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
end = time()
latency = end - start

# report performance
print('Accuracy: %.3f (%.3f) ' % (mean(scores), std(scores)))
print("\n Time to excute model: ",latency)

# Without any features selection method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standarization
sc=StandardScaler() 
# Set up the scaler just on the training set
sc.fit(X_train)
# Apply the scaler to the training and test sets
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
########################
# 2. Data Imbalance Methods

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['LOS_x_Short_Long'])

#2. ADASYN
#########
ada = ADASYN(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns
ada_data_X,ada_data_y=ada.fit_sample(X_train, y_train)
ada_data_X = pd.DataFrame(data=ada_data_X,columns=columns )
ada_data_y= pd.DataFrame(data=ada_data_y,columns=['LOS_x_Short_Long'])

##################################

X = data.loc[:, data.columns != 'LOS_x_Short_Long']
y = data.loc[:, data.columns == 'LOS_x_Short_Long']

tl = TomekLinks()
             
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns
tl_data_X,tl_data_y=tl.fit_sample(X_train, y_train)
tl_data_X = pd.DataFrame(data=tl_data_X,columns=columns )
tl_data_y= pd.DataFrame(data=tl_data_y,columns=['LOS_x_Short_Long'])

#ENN: EditedNearestNeighbours

X = data.loc[:, data.columns != 'LOS_x_Short_Long']
y = data.loc[:, data.columns == 'LOS_x_Short_Long']

enn = EditedNearestNeighbours()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns
enn_data_X,enn_data_y=enn.fit_sample(X_train, y_train)
enn_data_X = pd.DataFrame(data=enn_data_X,columns=columns )
enn_data_y= pd.DataFrame(data=enn_data_y,columns=['LOS_x_Short_Long'])

# SMOTE+ENN combination of over- and under-sampling

X = data.loc[:, data.columns != 'LOS_x_Short_Long']
y = data.loc[:, data.columns == 'LOS_x_Short_Long']

smote_enn = SMOTEENN(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns
smote_enn_data_X,smote_enn_data_y=smote_enn.fit_sample(X_train, y_train)
smote_enn_data_X = pd.DataFrame(data=smote_enn_data_X,columns=columns )
smote_enn_data_y= pd.DataFrame(data=smote_enn_data_y,columns=['LOS_x_Short_Long'])

# SMOTETomek combination of over- and under-sampling

X = data.loc[:, data.columns != 'LOS_x_Short_Long']
y = data.loc[:, data.columns == 'LOS_x_Short_Long']

smt = SMOTETomek(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns
smt_data_X,smt_data_y=smt.fit_sample(X_train, y_train)
smt_data_X = pd.DataFrame(data=smt_data_X,columns=columns )
smt_data_y= pd.DataFrame(data=smt_data_y,columns=['LOS_x_Short_Long'])

X_train, X_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(ada_data_X, ada_data_y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(tl_data_X, tl_data_y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(enn_data_X, enn_data_y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(smote_enn_data_X, smote_enn_data_y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(smt_data_X, smt_data_y, test_size=0.3, random_state=0)

#make prediction for selected models
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)

y_pred_baseline = baseline.predict(X_test)
y_proba_baseline = baseline.predict(X_test)#predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred_baseline))
# Classification report for Imbalanced
target_names = ['class 0', 'class 1'] # doctest : +NORMALIZE_WHITESPACE
#https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.classification_report_imbalanced.html#imblearn.metrics.classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred_baseline, target_names=target_names))

# Summary 
print('\n')
print('--------- Baseline Summary ------------')
print('\n')
print('Accuracy: {}'.format(round(accuracy_score(y_test, y_pred_baseline),2)))
print('Precision: {}'.format(round(precision_score(y_test, y_pred_baseline),2)))
print('Recall: {}'.format(round(recall_score(y_test, y_pred_baseline),2)))
print('F1-Score: {}'.format(round(f1_score(y_test, y_pred_baseline),2)))
print('AUC: {}'.format(round(roc_auc_score(y_test, y_proba_baseline),2)))

# Confusion Matrix for baseline
confusion_matrix = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

model=mdl

# Extract shap values
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
