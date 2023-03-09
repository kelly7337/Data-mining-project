from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier


data_pre = np.loadtxt('C:\\Users\\lenovo\\Desktop\\program2\\letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch)-ord('A')})

data_pre = pd.DataFrame(data_pre, columns=['string','int1','int2','int3','int4','int5','int6','int7','int8',
                                     'int9','int10','int11','int12','int13','int14','int15','int16'])

data = data_pre[data_pre['string'].isin([0,1])]
X1 = data[data.columns[1:17]]
Y = data[data.columns[0:1]]


pca = PCA(n_components=4)
X = pca.fit_transform(X1)

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.9,random_state=24)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

def SVM(x_train,x_test,y_train,y_test):
    C_range = [1,3,5,7,9]
    gamma_range = [2,4,6,8,10]
    param = dict(gamma=gamma_range, C=C_range)

    gc = GridSearchCV(svm.SVC(kernel='rbf'),param_grid=param,cv=5)
    y_train = y_train.values.ravel()
    gc.fit(x_train, y_train)

    print("Accuracy on the test set is:",gc.score(x_test,y_test))
    print("Best result in cross-validation:",gc.best_score_)
    print("The best model to choose is:",gc.best_estimator_)
    print("Results of each cross-validation for each hyperparameter is:",gc.cv_results_)

    for mean, params in zip(gc.cv_results_['mean_test_score'], gc.cv_results_['params']):
        print("%0.3f for %r" % (mean, params))

SVM(x_train,x_test,y_train,y_test)

# def Neural(x_train,x_test,y_train,y_test):
#     alpha = [0.01,0.001,0.0001]
#     learning_rate = ['constant', 'invscaling', 'adaptive']
#     param = dict(alpha=alpha, learning_rate=learning_rate)
#
#     gc = GridSearchCV(MLPClassifier(hidden_layer_sizes=(400,100),max_iter=500),param_grid=param,cv=5)
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
# Neural(x_train,x_test,y_train,y_test)




