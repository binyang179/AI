from PIL import Image
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import time

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
ALL_SET_LEN = 36

error_time = 0
correct_time = 0


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
# 警告： 转为灰度图 虽然对识别没有帮助 但是方便建模 否则后面会提示三维数组无法被塞进去二维数组
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def get_name_and_image():
    pic_path = 'C:/Users/Louis Song/Desktop/LetsFuckDog/data/captcha_pic/'
    all_image = os.listdir(pic_path)
    random_file = random.randint(0, 11313)  # 我当时样本有一万多 样本少了效果很不好
    base = os.path.basename(pic_path + all_image[random_file])
    name = os.path.splitext(base)[0]

    middle_image = Image.open(pic_path + all_image[random_file])
    # print(middle_image.shape)
    image = np.array(middle_image)
    # magicc_bug=image.shape
    # print(magicc_bug)
    # print(image.shape)
    return name.lower(), image


def name2vec(name):
    print(name)
    vector = np.zeros(MAX_CAPTCHA * ALL_SET_LEN)
    for i, c in enumerate(name):
        print(i, c)
        # 新增矩阵4*36
        # 如果是数字 那么
        if 47 < ord(str(c)) < 58:
            idx = i * 36 + ord(c) - 48
        else:
            idx = i * 36 + ord(c) - 97 + 10
        print(idx)
        vector[idx] = 1
    return vector


def vec2name(vec):
    name = []
    for i in range(0, len(vec), 36):
        b = vec[i:i + 36]
        # 数字转换
        if b[0] > 0:
            name.append('z')
        for count in range(1, 10):
            print(b)
            print(count)
            if b[count] > 0:
                name.append(str(count))
        for count in range(10, 36):
            if b[count] > 0:
                a = chr(int(count) + 97 - 10)
                name.append(a)
    return "".join(name)


# for i in range(10):
#     namelist, image_list = get_name_and_image()
#     print('-----------开始输出名字-----------')
#     print(namelist)
#     my_vec = name2vec(namelist)
#     print('---------返回结果----------')
#     print(vec2name(my_vec))


# 生成一个训练batch
# 注意 我修改了batch 原先值为64 现在扩大一倍 128
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * ALL_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        ''' 获取一张图，判断其是否符合（60，160，3）的规格'''
        while True:
            text, image = get_name_and_image()
            if image.shape == (60, 160, 3):  # 此部分应该与开头部分图片宽高吻合
                return text, image

    for i in range(batch_size):
        name, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        # middle = image.flatten()
        # batch_x[i, :] = 1*(image.flatten())
        # 这里 提示一个错误：ValueError: could not broadcast input array from shape (28800) into shape (9600)
        # 我怀疑错误原因是 少了一个维度
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


####################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * ALL_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    # 注意全连接层中，我们的图片40*160已经经过了3层池化层，也就是长宽都压缩了8倍，得到5*20大小。
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * ALL_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * ALL_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


#
#
# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, ALL_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, ALL_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            # 这里我修正了为0.35 进行尝试 不能让机器记那么多 否则我的样本就白费了
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 0.75})
                print(step, acc)
                # 如果准确率大于99%,保存模型,完成训练
                if acc > 0.99:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break

            step += 1


# train_crack_captcha_cnn()

# 训练完成后#掉train_crack_captcha_cnn()，取消下面的注释，开始预测，注意更改预测集目录

def crack_captcha(file_name):
    n = 1
    while n <= 10:
        middle_image = Image.open(file_name)
        image = np.array(middle_image)
        if image.shape != (60, 160, 3):
            print('原始图片错误，请核查')
            return "原始图片错误，请核查"
            pass
        else:
            image = convert2gray(image)
        # middle=image.flatten() /255
        try:
            image = image.flatten() / 255
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, ALL_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTCHA * ALL_SET_LEN)
            i = 0
            for n in text:
                vector[i * ALL_SET_LEN + n] = 1
                i += 1

            print(vector)
            predict_text = vec2name(vector)
            print(predict_text)
            return predict_text
        except TypeError as e:
            print(e)
            n += 1
            print(n)
            break


output = crack_captcha_cnn()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.'))