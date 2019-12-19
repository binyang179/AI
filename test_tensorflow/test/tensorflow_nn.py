import tensorflow as tf
a=tf.get_variable(name="first",shape=[3,2])
b=tf.get_variable(name="second",shape=[3,2])
c=a+b

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))

