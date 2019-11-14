#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import vgg16
import utils
from Nclasses import labels

img_path = input('Enter the picture path: ')
img = utils.load_image(img_path)

fig = plt.figure(u'Top-5 predictions')

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.forward(x)
    prob = sess.run(vgg.prob, feed_dict={x:img})
    # reverse sort
    top5 = np.argsort(prob[0])[-1:-6:-1] # the last -1 is strides
    # print('top5:', top5)

    values = []
    bar_label = []
    for n, i in enumerate(top5):
        values.append(prob[0][i])
        bar_label.append(labels[i])
        print(i, ':', labels[i], '----', utils.percent(prob[0][i]))

    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc ='g')
    ax.set_ylabel(u'probability')
    ax.set_title(u'Top-5')
    for a, b in  zip(range(len(values)), values):
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)
    plt.show()
