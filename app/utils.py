import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from PIL import Image, ImageFont, ImageDraw
from .consts import __chinese_word_count, __checkpoint_dir


# 构建一个三个卷积(3x3) + 三个最大池化层(2x2) +  两个FC层
def buildCnn(top_k):
    # with tf.device('/cpu:0'):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')  # image_size 64x64
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')  # image_size 62x62
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')  # image_size 31x31
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')  # image_size 29x29
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')  # image_size 15x15
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')  # image_size 13x13
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')  # image_size 7x7

    flatten = slim.flatten(max_pool_3)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh,
                               scope='fc1')  # 激活函数tanh
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), __chinese_word_count, activation_fn=None,
                                  scope='fc2')  # 无激活函数
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))  # softmax
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))  # 计算准确率

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)  #
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss,
                                                                   global_step=global_step)  # 自动调节学习率的随机梯度下降算法训练模型
    probabilities = tf.nn.softmax(logits)  #

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {
        'images': images,
        'labels': labels,
        'keep_prob': keep_prob,
        'top_k': top_k,
        'global_step': global_step,
        'train_op': train_op,
        'loss': loss,
        'accuracy': accuracy,
        'accuracy_top_k': accuracy_in_top_k,
        'merged_summary_op': merged_summary_op,
        'predicted_distribution': probabilities,
        'predicted_index_top_k': predicted_index_top_k,
        'predicted_val_top_k': predicted_val_top_k
    }


def predictPrepare():
    sess = tf.Session()
    graph = buildCnn(top_k=3)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(__checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    return graph, sess


def imagePrepare(image_path):
    temp_image = Image.open(image_path).convert('L')
    temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    return temp_image


def createImage(predword, imagepath):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fonts = ImageFont.truetype("./app/static/fonts/msyh.ttc", 36, encoding='utf-8')
    dr.text((15, 10), predword, font=fonts, fill="#000000")
    im.save(imagepath)
