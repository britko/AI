import tensorflow as tf

#행렬
a = tf.zeros([2, 10])
print(a)

#reduce 함수
b = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(tf.reduce_sum(b))
print(tf.reduce_mean(b))
print(tf.reduce_max(b))
print(tf.reduce_min(b))

#브로드캐스팅
c = b + 5
print(c)

#행렬 연산
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[10, 11],[21, 22], [30, 31]])
c = tf.matmul(a, b)
print(c)