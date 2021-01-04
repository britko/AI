import tensorflow as tf

a = tf.constant([[1,1],[2,2],[3,3]])        #3*2
b = tf.constant([[10,10],[20,20],[30,30]])  #3*2

#concat: 텐서 합치기 (0: 행 기준 합치기, 1: 열 기준 합치기)
c = tf.concat([a,b],0)
print(c)
d = tf.concat([a,b],1)
print(d)

#stack: 지정하는 차원으로 확장하여 텐서를 쌓아줌 (0: 행 기준 쌓기, 1: 열 기준 쌓기)
e = tf.stack([a,b],0)
print(e)
f = tf.stack([a,b],1)
print(f)