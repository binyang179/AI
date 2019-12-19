import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(0, 6, 0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)
# plt.plot(x, y1, label="sin")
# plt.plot(x, y2, linestyle = "--", label='cos')
# plt.xlabel("x")
# plt.ylabel('y')
# plt.title('sin & cos')
# plt.legend()
# plt.show()

#sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#ReLU function
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
print(x)
y = sigmoid(x)
# y = relu(x)

plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
plt.show()
print(sigmoid(x))


def mean_squard_error(y, t):
    return 0.5 * np.sum((y - t)**2)


