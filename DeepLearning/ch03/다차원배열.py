import numpy as np

print("-----------1차원-----------")

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))   #배열의 차원 수
print(A.shape)  #배열의 형상
print(A.shape[0])

print("-----------2차원-----------")

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)