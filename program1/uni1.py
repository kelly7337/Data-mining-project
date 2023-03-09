import numpy as np
from matplotlib import pyplot as plt
import xlrd

file_location = "C:\\Users\\lenovo\\Desktop\\program1\\Concrete_Data.xls"
data = xlrd.open_workbook(file_location)

sheet = data.sheet_by_index(0)

x_data = [sheet.cell_value(r, 0) for r in range(1, 901)]
y_data = [sheet.cell_value(r, 8) for r in range(1, 901)]


lr = 0.000001
theat0 = 0
theat1 = 0
iteators = 1000


def compute_loss(x_data, y_data, theat0, theat1):
    sum = 0

    for i in range(len(x_data)):
        sum += pow((theat0 + theat1 * x_data[i]) - y_data[i], 2)

    return sum / (2 * len(x_data))


def gradient_fun(x_data, y_data, theat0, theat1, iteators, lr):
    m = float(len(x_data))

    cost = np.zeros(iteators)

    for j in range(iteators):
        sum1 = 0
        sum2 = 0
        for i in range(len(x_data)):
            sum1 += (theat0 + theat1 * x_data[i]) - y_data[i]
            sum2 += ((theat0 + theat1 * x_data[i]) - y_data[i]) * x_data[i]

        theat0 = theat0 - lr * sum1 / m
        theat1 = theat1 - lr * sum2 / m
        cost[j] = compute_loss(x_data, y_data, theat0, theat1)

        print('iteators = ', j, 'loss =', compute_loss(x_data, y_data, theat0, theat1), 'theat0 = ', theat0,
              'theat1 = ', theat1)

    return theat0, theat1, cost


theat0, theat1, cost = gradient_fun(x_data, y_data, theat0, theat1, iteators, lr)

#plt.plot(x_data, y_data, 'b.')
#plt.plot(x_data, theat0 + theat1 * x_data, 'r')
#plt.show()
a = [(i * theat1 + theat0) for i in x_data]
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, a, 'r')
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iteators), cost, 'b')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('cost__iteators')
plt.show()


file_location = "C:\\Users\\lenovo\\Desktop\\program1\\Concrete_Data.xls"
data = xlrd.open_workbook(file_location)

sheet = data.sheet_by_index(0)

x_data1 = [sheet.cell_value(r, 0) for r in range(901, 1031)]


def predict_func(x_data1, theat0, theat1):
    theat0 = 0.0019185439886793712
    theat1 = 0.11983542376245027
    pre_y = np.zeros(len(x_data1))
    for i in range(len(x_data1)):
        pre_y[i] = theat0 + theat1 * x_data1[i]
    print(compute_loss(x_data1, pre_y, theat0, theat1))

    a = [(i * theat1 + theat0) for i in x_data1]
    plt.plot(x_data1, pre_y, 'b.')
    plt.plot(x_data1, a, 'r')
    plt.show()


#theat0 = 0.03501490948451012
#theat1 = 1.4788038980136102
# predict_func(x_data1, theat0, theat1)
