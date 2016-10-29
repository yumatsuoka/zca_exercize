# coding: utf-8
# # ZCAの実装とその動作の検証 with CIFAR-10

from __future__ import print_function

import six
import numpy
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import cifar10


print("集中度推定画像群の学習用画像群を読み込む")
dataset = cifar10.load()
train_data_dict = {'data':dataset['train']['data'].astype(numpy.float32)}


# PCAした結果を返す関数
def principal_components(x):
    x = x.transpose(0, 2, 3, 1)
    flatx = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = numpy.dot(flatx.T, flatx) / flatx.shape[1]
    #U, S, V = linalg.svd(sigma)
    U, S, V = numpy.linalg.svd(sigma)
    eps = 0.0001 #kerasでは10e-7
    return numpy.dot(numpy.dot(U, numpy.diag(1. / numpy.sqrt(S + eps))), U.T)


print("principal_componentsをデータセットから作成")
pc = principal_components(train_data_dict['data'])


# ZCAを入力データに適用した出力を返す関数
def zca_whitening(x, principal_components):
    x = x.transpose(1,2,0)
    flatx = numpy.reshape(x, (x.size))
    whitex = numpy.dot(flatx, principal_components)
    x = numpy.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x


print("ZCAを実行")
N = 100
test_imgs = train_data_dict['data'][:N]
app_zca_imgs = numpy.asarray([zca_whitening(test_imgs[idx], pc) 
    for idx in six.moves.range(N)])

#print("output imgs")
#output_imgs = numpy.c_[test_imgs.transpose(0, 2, 3, 1), app_zca_imgs]
#print("output_imgs.shape", output_imgs.shape)

"""
output_imgs = test_imgs.transpose(0, 2, 3, 1)
print(output_imgs[0][20])
#output_imgs = test_imgs.reshape(test_imgs.shape[0], 32, 32, 3)
r_img = [Image.fromarray(numpy.uint8(output_imgs[idx])) for idx in six.moves.range(10)]
for i in range(10):
    r_img[i].save("rr_img_{}.png".format(i))
"""

print("plot raw_imgs")
plt.clf()
output_imgs = test_imgs.transpose(0, 2, 3, 1) / 255.
print('output_imgs.shape', output_imgs.shape)
fig = plt.figure(figsize=(100, 100))
for i in six.moves.range(len(output_imgs)):
    ax = fig.add_subplot(10, len(output_imgs)/10, i+1)
    plt.axis("off")
    plt.imshow(output_imgs[i], interpolation='nearest', cmap='Greys_r')
plt.savefig("raw_imgs.png")

print("plot zca_whitening imgs")
plt.clf()
#output_imgs = numpy.uint8(app_zca_imgs) + 128
print("処理前",output_imgs[0])
output_imgs = numpy.round(numpy.abs(app_zca_imgs))*127 +128
print("処理後", output_imgs[0])
fig = plt.figure(figsize=(100, 100))
for i in six.moves.range(len(output_imgs)):
    ax = fig.add_subplot(10, len(output_imgs)/10, i+1)
    plt.axis("off")
    plt.imshow(output_imgs[i], interpolation='nearest', cmap='Greys_r')
plt.savefig("imgs_applied_zca.png")
