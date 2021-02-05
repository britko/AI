import numpy as np
import matplotlib.pyplot as plt

#데이터 준비
x = np.arange(0, 6, 0.1)    #0에서 6까지 0.1간격(그래프의 정확도)으로 생성
y = np.sin(x)   #numpy의 sin함수(x에서의 sin값)

#그래프 그리기
plt.plot(x, y)
plt.show()