import tensorflow as tf

import data_reader


def inference(input_data):
    """
    正向传播计算
    :return:
    """
    input_shape = input_data.shape
    weights = tf.get_variable(name="weights",
                              shape=[input_shape[1], 1],
                              initializer=tf.initializers.random_normal)
    biases = tf.get_variable(name="biases",
                             shape=[1],
                             initializer=tf.initializers.constant)
    output = tf.matmul(input_data, weights) + biases
    return tf.nn.sigmoid(output)


def train(input_x, input_y, ephocs=10, batch_size=100):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")
    output = inference(x)
    # 定义损失函数
    cost = -tf.reduce_sum(y * tf.log(output) + (1 - y) * tf.log(1 - output))  # 逻辑回归的损失函数
    entry_cost = tf.train.GradientDescentOptimizer(0.0003).minimize(cost)
    batches = input_x.shape[0] // batch_size
    if input_x.shape[0] % batch_size != 0:
        batches += 1
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for _ in range(ephocs):
            for batch in range(batches):
                start = batch * batch_size % input_x.shape[0]
                end = min(start + batch_size, input_x.shape[0])
                sess.run([entry_cost], feed_dict={x: input_x[start:end], y: input_y[start:end]})
            c = sess.run([cost], feed_dict={x: input_x, y: input_y})
            print(c)


data, label =  data_reader.read()
train(data, label)
