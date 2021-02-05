#Relu 클래스는 mask라는 인스턴스 변수를 가집니다.
#mask는 True/False로 구성된 넘파이 배열로, 
# 순전파의 입력인 x의 원소 값이 0이하인 인덱스는 True, 
# 그 외(0보다 큰 원소)는 False로 유지합니다.
class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        sekf.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

#예시
import numpy as np

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)

#순전파 때의 입력 값이 0 이하면 역전파 때의 값은 0이 돼야 합니다.
#그래서 역전파 때는 순전파 때 만들어둔 mask를 써서 
# mask의 원소가 True인 곳에는 상류에서 전파된 dout을 0으로 설정합니다.