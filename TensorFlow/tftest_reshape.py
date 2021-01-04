import tensorflow as tf

a = tf.constant([[1,2,3],[10,11,12]])
print(a)

x = tf.reshape(a, [3,2])
print(x)

#자료형 변경
c = tf.cast(a, tf.float32)
print(c)


#부호 반전 tf.negative
y = tf.constant(-3)
print(tf.negative(y))