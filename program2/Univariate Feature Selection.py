from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,f_classif,chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

data_pre = np.loadtxt('C:\\Users\\lenovo\\Desktop\\program2\\letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch)-ord('A')})

data_pre = pd.DataFrame(data_pre, columns=['string','int1','int2','int3','int4','int5','int6','int7','int8',
                                     'int9','int10','int11','int12','int13','int14','int15','int16'])

data = data_pre[data_pre['string'].isin([0,1])]
X1 = data[data.columns[1:17]]
Y = data[data.columns[0:1]]

X = SelectKBest(chi2, k=4).fit_transform(X1,Y)
#print(X)

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.9,random_state=24)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# def DT(x_train,x_test,y_train,y_test):
#     max_range = [3,4,5,6,7]
#     random_range = [1,4,6,7,9]
#     param = dict(max_depth=max_range, random_state=random_range)
#
#     gc = GridSearchCV(DecisionTreeClassifier(criterion='gini'),param_grid=param,cv=5)
#     y_train = y_train.values.ravel()
#     gc.fit(x_train, y_train)
#
#     print("Accuracy on the test set is:",gc.score(x_test,y_test))
#     print("Best result in cross-validation:",gc.best_score_)
#     print("The best model to choose is:",gc.best_estimator_)
#     print("Results of each cross-validation for each hyperparameter is:",gc.cv_results_)
#
#     for mean, params in zip(gc.cv_results_['mean_test_score'], gc.cv_results_['params']):
#         print("%0.3f for %r" % (mean, params))
#
#
# DT(x_train,x_test,y_train,y_test)

def RandomForest(x_train,x_test,y_train,y_test):
    param = {"n_estimators":[100,110,120,130,140],"max_depth":[3,4,5,6,7]}

    gc = GridSearchCV(RandomForestClassifier(),param_grid=param,cv=5)
    y_train = y_train.values.ravel()
    gc.fit(x_train, y_train)

    print("Accuracy on the test set is:",gc.score(x_test,y_test))
    print("Best result in cross-validation:",gc.best_score_)
    print("The best model to choose is:",gc.best_estimator_)
    print("Results of each cross-validation for each hyperparameter is:",gc.cv_results_)

    for mean, params in zip(gc.cv_results_['mean_test_score'], gc.cv_results_['params']):
        print("%0.3f for %r" % (mean, params))

RandomForest(x_train,x_test,y_train,y_test)

