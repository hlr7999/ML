#coding:utf-8
from skimage import io, transform
import matplotlib.pyplot as plt
from pylab import mpl
# to show Chinese and +- at the picture
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def load_image(path):
    try:
        img = io.imread(path)
        img = img / 255.0
    except:
        print('Cannot open ' + path);
        exit(-1)
    fig = plt.figure("Centre and Resize")
    # split the figure to 3 parts
    # the first is original picture
    fig0 = fig.add_subplot(1, 3, 1) # params can also be 131
    fig0.set_xlabel(u'Original Picture')
    fig0.imshow(img)
    # the second is centre picture
    short_edge = min(img.shape[:2])
    y = (img.shape[0] - short_edge) // 2
    x = (img.shape[1] - short_edge) // 2
    cen_img = img[y:y+short_edge, x:x+short_edge]
    fig1 = fig.add_subplot(132)
    fig1.set_xlabel(u'Centre Picture')
    fig1.imshow(cen_img)
    # the third is resize picture
    re_img = transform.resize(cen_img, (224, 224))
    fig2 = fig.add_subplot(1, 3, 3)
    fig2.set_xlabel(u'Resize Picture')
    fig2.imshow(re_img)
    # reshape the img to test
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

def percent(value):
    return '%.2f%%' % (value * 100)
