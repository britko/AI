import numpy as np
import matplotlib.pyplot as plt

#데이터
x = np.arange(-5.0, 5.0, 0.1)
h = 1/(1+np.exp(-x))    #sigmoid 함수 식

#그래프
plt.plot(x, h, color='r', label="sigmoid")  #색:빨간색, 이름:sigmoid
plt.xlabel("x")
plt.ylabel("h")
plt.ylim(-0.1, 1.1)
plt.title("sigmoid function")
plt.legend()    #그래프 범례 추가
plt.show()