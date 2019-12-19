import tensorflow as tf
import os
from input_bank_data import read_data


def get_weight(weight_shape, regularization=None):
    """
    获取权重信息
    :param weight_shape:权重的大小，该参数为矩阵，例如[5,4]
    :param regularization:用于对权重进行正则化的函数
    :return:根据权重大小创建的权重列表
    """
    weights = tf.get_variable("weights",
                              shape=weight_shape,
                              dtype="float",
                              initializer=tf.initializers.random_normal)
    if regularization:
        loss = regularization(weights)
        tf.add_to_collection("global_losses", loss)
    return weights


def get_biases(biases_shape):
    """
    获取权重信息
    :param biases_shape:权重信息的大小，该参数是一个矩阵，如[2],[3,2]
    :return:根据权重大小创建的矩阵信息
    """
    biases = tf.get_variable("biases",
                             shape=biases_shape,
                             dtype="float",
                             initializer=tf.initializers.constant)
    return biases


def inference(input_tensor):
    """
    正向传播计算
    :param input_tensor:输入信息
    :return:
    """
    with tf.variable_scope("first_nn", reuse=tf.AUTO_REUSE):
        weights = get_weight(weight_shape=[input_tensor.shape[1], 16],
                             regularization=tf.contrib.layers.l2_regularizer)
        biases = get_biases(biases_shape=[1])
        first_result = tf.nn.sigmoid(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope("second_nn", reuse=tf.AUTO_REUSE):
        weights = get_weight(weight_shape=[16, 2],
                             regularization=tf.contrib.layers.l2_regularizer)
        biases = get_biases(biases_shape=[1])
        second_result = tf.nn.sigmoid(tf.matmul(first_result, weights) + biases)
    return second_result


def train(input_x, input_y):
    """
    训练模型
    :param input_x:用户的输入训练数据，为一个np.ndarray
    :param input_y:用户的输入标签，为一个np.ndarray
    :return:None
    """
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")

    y_ = inference(x)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    cross_entropy = tf.train.AdamOptimizer(0.006).minimize(entropy)

    # 验证
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 计算整体的batch
        batch_size = 1024
        length = input_x.shape[0]
        total_batch = length // batch_size if length % batch_size == 0 else length // batch_size + 1
        for i in range(0, 10):
            for b in range(0, total_batch):
                start = batch_size * b % length
                end = min(start + batch_size, length)
                sess.run(cross_entropy, feed_dict={x: input_x[start:end], y: input_y[start:end]})
            acc = sess.run(accuracy, feed_dict={x: input_x, y: input_y})
            saver.save(sess, os.path.join("bank_models", "models.ckpt"))
            print("第%d次之后的accuracy=%f" % (i, acc))


def validator(input_x, input_y):
    """
    对模型进行测试
    :param input_x:输入的测试样本，为np.ndarray
    :param input_y:输入的测试样本的标签，为np.ndarray
    :return:None
    """
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")

    y_ = inference(x)
    # 验证
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), "float"))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("bank_models")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(sess.run([accuracy], feed_dict={x: input_x, y: input_y}))


# if __name__ == "__main__":
#    train_x, train_y, test_x, test_y = read_data(os.path.join("data", "bank-full.csv"))
#    train(train_x, train_y)


if __name__ == "__main__":
    _, _, test_x, test_y = read_data(os.path.join("data", "bank-full.csv"))
    validator(test_x, test_y)