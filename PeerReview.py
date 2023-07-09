#import libraries
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score,confusion_matrix

'''
In this Peer review project I will be using "Palmer Archipelago (Antarctica) penguin dataset"

Overview of dataset:
Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

features:
culmen_length_mm: culmen length (mm) - (float64)
culmen_depth_mm: culmen depth (mm) - (float64)
flipper_length_mm: flipper length (mm) - (float64)
body_mass_g: body mass (g) - (float64)
island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica) - (Object)
species: penguin species (Chinstrap, Adélie, or Gentoo) (Object)

outcome variable:
sex: penguin sex - (Object)


with above mentioned features I want to train machine learning model which will predict sex of the penguin

'''
#data import and quick sample of data
data = pd.read_csv('./data/penguins.csv')

print(data.head())

print('\n')
print(data.dtypes)

print('\n')
print(data.species.value_counts())

print('\n')
print(data.island.value_counts())

print('\n')
print(data.sex.value_counts())

print('\n')
print(data.isna().sum())

print('\n')
print(data.describe())


'''
step 1 - Exploritory Data Analysis

By exmaining data types we can see that some of the feature columns are not numeric type
for example

First we have to drop year and rowid columns since it will not provide any siginificance for this analysis. We also have to drop all the N/A rows from our outcome (Sex) variable.

In this porject I will use lambda to convert sex into binary M=1, F=0 and get dummies to convert other object type features into numeric type
after that we have to Scale all the float variables and we will use standard scaler for that.

After conversion I will use train test split to split my dataset into training data and testing data of 7:3 ratio
'''

#Droping Year and rowid column
data = data.drop(['year','rowid'],axis=1)

#Droping N/A values from Sex

data.dropna(subset=['sex'],inplace=True)

#Converting Sex into Binary
data['sex'] = data.sex.apply(lambda x: 1 if x=="male" else 0)

#data = data.drop('species',axis=1)

# One Hot encoding other columns
labelColumns = data.select_dtypes(include=[np.object_]).columns.to_list()
data = pd.get_dummies(data=data,columns=labelColumns,dtype=np.int32)


# Scaling Data

x = data.drop('sex',axis=1)
y = data['sex']
sc = StandardScaler()
#mn = MinMaxScaler()
x_scaled = sc.fit_transform(x)
#x_scaled = mn.fit_transform(x)
x = pd.DataFrame(x_scaled, index= x.index, columns=x.columns)
#spliting data using train test split




x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


def getPerformaceMetrics(y_true,y_pred,label):
    return pd.Series({'accuracy': accuracy_score(y_true, y_pred),
                      'precison': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred)},name=label)


# End of Preprocessing

#Now we will use Logistic Regression, Decision Trees and Random Forest to predict outcome variable and then compare results


#Logistic Regression without tuning any hyper-parameters

lr = LogisticRegression(random_state=42)
lr.fit(x_train,y_train)
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

train_test_full_error_lr = pd.concat([getPerformaceMetrics(y_train,y_train_pred,'Train'),
                                      getPerformaceMetrics(y_test,y_test_pred,'Test')],axis=1)

print('\nLogistic Regression (Without Hyperparameters):')
print(train_test_full_error_lr)

#Logistic Regression with GridSearchCV

param_grid = {'solver': ['saga'],
                'penalty':['l1','l2'],
                'C':[0.01, 0.1, 1.0, 10.0, 100.0]}

GR_LR = GridSearchCV(LogisticRegression(random_state=42,max_iter=10000),param_grid=param_grid,scoring='accuracy',n_jobs=-1)
check = os.path.isfile('./Models/GR_LR.model')

if check == True:
    GR_LR = pickle.load(open('./Models/GR_LR.model','rb'))
else:
    GR_LR.fit(x_train,y_train)
    pickle.dump(GR_LR,open('./Models/GR_LR.model','wb'))

print('\n')
print(GR_LR.best_estimator_)
y_train_pred = GR_LR.predict(x_train)
y_test_pred = GR_LR.predict(x_test)


train_test_full_error_lrcv = pd.concat([getPerformaceMetrics(y_train,y_train_pred,'Train'),
                                      getPerformaceMetrics(y_test,y_test_pred,'Test')],axis=1)

print('\nLogistic Regression (With Hyperparameters):')
print(train_test_full_error_lrcv)

#DecisionTreeClassifier

dt =DecisionTreeClassifier(random_state=42)

dt.fit(x_train,y_train)

y_train_pred = dt.predict(x_train)
y_test_pred = dt.predict(x_test)

train_test_full_error_dt = pd.concat([getPerformaceMetrics(y_train,y_train_pred,'Train'),
                                      getPerformaceMetrics(y_test,y_test_pred,'Test')],axis=1)

print('\nDecision Trees (Without Hyperparameters):')
print(train_test_full_error_dt)

check = os.path.isfile('./Model/GR_DT.model')

param_grid = {'max_depth':range(1,dt.tree_.max_depth+1,2),
              'max_features':range(1, len(dt.feature_importances_) + 1),
              'min_samples_split':range(1,11),
              'min_samples_leaf':range(1,10)}
#[1,10,25,50,100,200,500]
GR_DT = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid=param_grid,scoring='accuracy',n_jobs=-1)

if check == True:
    GR_DT = pickle.load(open('./Model/GR_DT.model','rb'))
else:
    GR_DT.fit(x_train,y_train)
    pickle.dump(GR_DT,open('./Models/GR_DT.model','wb'))


print('\n')
print(GR_DT.best_estimator_)
y_train_pred = GR_DT.predict(x_train)
y_test_pred = GR_DT.predict(x_test)


train_test_full_error_dtcv = pd.concat([getPerformaceMetrics(y_train,y_train_pred,'Train'),
                                      getPerformaceMetrics(y_test,y_test_pred,'Test')],axis=1)

print('\nDecision Trees (With Hyperparameters):')
print(train_test_full_error_dtcv)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

rf.fit(x_train,y_train)

y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)

train_test_full_error_rf = pd.concat([getPerformaceMetrics(y_train,y_train_pred,'Train'),
                                      getPerformaceMetrics(y_test,y_test_pred,'Test')],axis=1)

print('\nRandom Forest (Without Hyperparameters):')
print(train_test_full_error_rf)

param_grid = {'max_depth':range(1,dt.tree_.max_depth+1,2),
              'max_features':range(1, len(dt.feature_importances_) + 1),
              'criterion':['gini','entropy'],
              'n_estimators':[100,200,400]}
#[1,10,25,50,100,200,500]
GR_RF = GridSearchCV(RandomForestClassifier(random_state=42),param_grid=param_grid,scoring='accuracy',n_jobs=-1)

check = os.path.isfile('./Models/GR_RF.model')

if check == True:
    GR_RF = pickle.load(open('./Models/GR_RF.model','rb'))
else:
    GR_RF.fit(x_train,y_train)
    pickle.dump(GR_RF,open('./Models/GR_RT.model','wb'))


print('\n')
print(GR_RF.best_estimator_)
y_train_pred = GR_RF.predict(x_train)
y_test_pred = GR_RF.predict(x_test)


train_test_full_error_rfcv = pd.concat([getPerformaceMetrics(y_train,y_train_pred,'Train'),
                                      getPerformaceMetrics(y_test,y_test_pred,'Test')],axis=1)

print('\nDecision Trees (With Hyperparameters):')
print(train_test_full_error_rfcv)

sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True)
plt.show()
