import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

tf.print(node1, node2)
print(node1, node2)
print(tf.reduce_sum(tf.random.normal([1000, 1000])))