import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def process_data(path):
    # 特征值: 固定酸度、挥发性酸、柠檬酸、残留糖、氯化物、无二氧化硫、总二氧化硫、密度、ph值、硫酸盐、酒精、品质
    df = pd.read_csv(path, header=0, delimiter=";",
                     names=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                            "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
                            "quality"])
    # print(df.head())
    df = df.apply(lambda column: (column - column.mean()) / column.std())
    # print(df.head())
    df.insert(0, "bais", 1)
    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    # random_state  随机划分训练集和测试集时候，划分的结果并不是那么随机，也即，确定下来random_state是某个值后，重复调用这个函数，划分结果是确定的
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def test_case1():
    file = "winequality-red.csv"
    X_train, X_test, y_train, y_test = process_data(file)
    print("训练数据集:", X_train.shape, "\n", X_train[:2])
    print("测试数据集:", X_test.shape, "\n", X_test[:2])
    print("训练集标签:", y_train.shape, "\n", y_train[:2])
    print("测试集标签:", y_test.shape, "\n", y_test[:2])


# if __name__ == "__main__":
#     test_case1()


def lr_cost(theta, X, y):
    m = X.shape[0]  # x.shape获取X的大小，一般为一个tuple(m,n)，其中m指行   #数，n指列数。这里我们通过下表0取行数
    inner = np.dot(X, theta) - y  # 数据集X 和theta内积运算，符合矩阵乘法运算法则。#其结果取X的行数，theta的列数。然后将结果与y相减，得到误差
    square_sum = np.dot(inner.T, inner)  # inner.T表示将inner转置
    cost = square_sum / (2 * m)
    return cost


def gradient(theta, X, y):
    m = X.shape[0]  # x.shape获取X的大小，一般为一个tuple(m,n)，其中m指行   #数，n指列数。这里我们通过下表0取行数
    inner = np.dot(X.T, (np.dot(X, theta) - y))
    return inner / m  # 求导过程中，分母2m被约掉了，m仍然存在，所以分母为m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    initial_cost = lr_cost(theta, X, y)
    print("inital cost:%.2f"%initial_cost)
    cost_data = [initial_cost]  # 声明列表来保存迭代过程中代价函数的值
    _theta = theta.copy()  # 拷贝theta，避免和原来的theta混淆
    for i in range(1,epoch):  # epoch为迭代的批次，一批次包含全部数据的训练。例如有100万条数据，每次取10000条，要循环100次才能取完。那么一批就包含这#100次的运算。
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_value = lr_cost(_theta, X, y)
        if i % 100 == 0:
            print("第%d个100次迭代,cost=%.2f"%(int(i/100), cost_value))
        cost_data.append(cost_value)  # 保存迭代过程中代价函数值
    return _theta, cost_data


def predict_evaluate(theta, X_test, y_test):  # theta值是优化后的值
    y_p = np.dot(X_test, theta.T)
    m = len(X_test)
    return  ((y_p - y_test) ** 2).sum()/m


def test_case2():
    file= "winequality-red.csv"
    X_train, X_test, y_train, y_test =  process_data(file)
    alpha = 0.01
    theta = np.zeros(X_train.shape[1])
    epoch = 5000
    final_theta, _= batch_gradient_decent(theta, X_train, y_train, epoch, alpha=alpha)
    print("final_theta:\n", final_theta)
    mse = predict_evaluate(final_theta , X_test ,y_test)
    print("mse:",mse)

# if __name__ == "__main__": #编写主函数
#     #test_case1()调用函数test_case1
#     test_case2()


def plot_cost_data(costs):
    sns.tsplot(time=np.arange(len(costs)), data=costs)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('cost', fontsize=18)
    plt.savefig("epoch_cost.png")
    plt.show()


def plot_learning_rate(X, y, epoch=500):
    base = np.logspace(-1, -5, num=3)
    candidate = np.sort(np.concatenate((base, base * 3)))
    print(candidate)
    theta = np.zeros(X.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha in candidate:
        _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
        ax.plot(np.arange(epoch), cost_data, label=alpha)
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('cost', fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    ax.set_title('learning rate', fontsize=18)
    plt.savefig("epoch_cost_rate.png")
    plt.show()


def test_case3():
    file = "winequality-red.csv"
    X_train, X_test, y_train, y_test = process_data(file)
    alpha = 0.01
    theta = np.zeros(X_train.shape[1])
    epoch = 500
    final_theta, cost_data = batch_gradient_decent(theta, X_train, y_train, epoch, alpha=alpha)
    plot_cost_data(cost_data)
    plot_learning_rate(X_train, y_train)


if __name__ == '__main__':
    test_case3()