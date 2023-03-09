import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data_pre = np.loadtxt('C:\\Users\\lenovo\\Desktop\\program2\\letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch)-ord('A')})

data_pre = pd.DataFrame(data_pre, columns=['string','int1','int2','int3','int4','int5','int6','int7','int8',
                                     'int9','int10','int11','int12','int13','int14','int15','int16'])

data = data_pre[data_pre['string'].isin([0,1])]

X = data[data.columns[1:17]]
Y = data[data.columns[0:1]]
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.9,random_state=24)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

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
