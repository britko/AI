import numpy as np
from common.functions import *


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 손실
        self.y = None   # softmax의 출력
        self.t = None   # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    #역전파 때는 전파하는 값을 배치의 수(batch_size)로 나눠서
    #데이터 1개당 오차를 앞 계층으로 전파한다.
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx