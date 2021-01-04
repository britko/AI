import tensorflow as tf

#행렬 연산
a = tf.constant([[1, 2], [4, 5]])
b = tf.constant([[10, 11],[21, 22]])
c = tf.matmul(a, b)     #행렬곱
d = tf.add(a, b)        #덧셈
e = tf.subtract(a, b)   #뺄셈
print(c)
print(d)
print(e)