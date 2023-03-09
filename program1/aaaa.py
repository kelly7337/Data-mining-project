import numpy as np
import pandas as pd



def grad(x, y, theat):
    G = []
    x = np.c_[np.ones(x.shape[0]), x]
    for j in range(len(theat)):
        gi = 0
        for i in range(len(x)):
            x_ij = x[(i), (j)]
            y_i = float(y[i])
            theat_0 = float(theat[0])
            theat_j = float(theat[j])
            y_hat = theat_0 + theat_j * x_ij
            gi += (y_hat - y_i) * x_ij * (1 / len(x))
        G.append(gi)
    G = np.array(G)
    G[np.isnan(G)] = 0
    return (G)


def grad_descent(x, y, lr, maxtime,error):
    i = 0
    x = np.array(x)
    y = np.array(y)
    theat = np.random.randint(-1, 1, size=(x.shape[1] + 1, 1))

    while (i < maxtime):
        i += 1
        theat_old = theat.T
        direct = lr * grad(x, y, theat)
        theat2 = theat.T - direct
        distant = np.sqrt(np.sum(direct ** 2))
        if distant < error:
            break
        theat = theat2.T
    print('number of iterations：%d' % i)
    print('last iteration error：%s' % distant)
    theat[np.isnan(theat)] = 0
    return theat


data = pd.read_csv('C:\\Users\\lenovo\\Desktop\\program1\\Concrete_Data.csv', encoding='utf-8')
x = data.iloc[1:901, 0:7]
y = data.iloc[1:901, 8]
Theat = grad_descent(x, y, 0.00001, 100, 5)

print(Theat)

import matplotlib.pyplot as plt

X = pd.DataFrame(np.c_[np.ones(x.shape[0]), x])[4]
Y = pd.DataFrame(np.array(data.iloc[1:901, 8]))[0]
Y_hat = pd.DataFrame(np.dot(np.c_[np.ones(x.shape[0]), x], Theat))[0]
# plt.scatter(X,Y)
# plt.scatter(X,Y_hat)

plt.plot(X, Y, 'b.')
plt.plot(X, Y_hat, 'r')
plt.show()


x1 = data.iloc[901:1031, 0:7]
x1 = np.array(x1)
def predict_func(x1, Theat, maxtime, Y, lr):
    pre_y = np.zeros(len(x1))
    for i in range(len(x1)):
        for j in range(7):
            pre_y[i] = Theat[i] * x1[(i), (j)]
    while (i < maxtime):
        i += 1
        theat_old = Theat.T
        direct = lr * grad(x, y, Theat)
        distant = np.sqrt(np.sum(direct ** 2))
        print (distant)


    import matplotlib.pyplot as plt
    X = pd.DataFrame(np.c_[np.ones(x.shape[0]), x])[4]
    Y = pd.DataFrame(np.array(data.iloc[:, 7]))[0]
    pre_y = pd.DataFrame(np.dot(np.c_[np.ones(x.shape[0]), x], Theat))[0]
    plt.scatter(X, Y)
    plt.scatter(X, pre_y)

# predict_func(x1, Theat,100, Y, 0.00001)

