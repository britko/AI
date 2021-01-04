import tensorflow as tf

a = tf.constant([[1,2,3],[4,5,6]])          #2*3
b = tf.constant([[10,11],[20,21],[30,31]])  #3*2
c = tf.slice(a, [1,0], [1,3])  #(행렬, 시작지점[행렬], 사이즈[행렬])

print(c)