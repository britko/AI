import numpy as np
import matplotlib.pyplot as plt

#softmax함수의 원형
def softmax_org(a):
    exp_a = np.exp(a)   #지수함수
    sum_exp_a = np.sum(exp_a)   #지수함수의 합
    y = exp_a / sum_exp_a   #지수함수/지수함수의 합

    return y

#오버플로우를 개선한 softmax함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   #오버플로우 대책: a[] - a
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
    
a = np.array([990, 1000, 1010])

print(softmax_org(a))
print(softmax(a))

plt.plot(softmax(a))
plt.show()