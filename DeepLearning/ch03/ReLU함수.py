import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x) #두 입력 중 큰 값을 선택해 반환하는 함수

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.title("ReLU function")
plt.show()