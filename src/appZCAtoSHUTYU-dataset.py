# coding: utf-8
# # ZCAの実装とその動作の検証

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy
from PIL import Image
import pandas as pd
from scipy import linalg


# CSVから画像を読み込む関数
def get_dataset(data_list):
    csv_data = pd.read_csv(data_list, header=None)
    data = numpy.asarray([numpy.asarray(Image.open(csv_data[0][i], 'r'))
        for i in range(len(csv_data))]).astype(numpy.float32)
    target = numpy.asarray([csv_data[1][i]
        for i in range(len(csv_data))]).astype(numpy.int32)
    data /= 255.
    # data = numpy.transpose(data, (0, 3, 1, 2))
    print("data.shape", data.shape)
    return data, target

# 集中度推定画像群の学習用画像群を読み込む
l_list = './target/train_190.csv'
train_d, train_t = get_dataset(l_list)
plt.imshow(train_d[0])


# PCAした結果を返す関数
def principal_components(x):
    flatx = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = numpy.dot(flatx.T, flatx) / flatx.shape[1]
    #U, S, V = linalg.svd(sigma)
    U, S, V = numpy.linalg.svd(sigma)
    return numpy.dot(numpy.dot(U, numpy.diag(1. / numpy.sqrt(S + 10e-7))), U.T)

# principal_componentsをデータセットから作成
pc = principal_components(train_d)
print("principal_components.shape", pc.shape)
#plt.imshow(pc)
plt.imshow(pc, interpolation='nearest', cmap='Greys_r')


# ZCAを入力データに適用した出力を返す関数
def zca_whitening(x, principal_components):
    flatx = numpy.reshape(x, (x.size))
    whitex = numpy.dot(flatx, principal_components)
    x = numpy.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

# ZCAを実行して，画像を減算して出力
ex_img = train_d[0]
app_zca_img = zca_whitening(ex_img, pc)
print("img.shape which is applied to ZCA", app_zca_img.shape)
#plt.imshow(app_zca_img)
#plt.imshow(app_zca_img, cmap='Greys_r')
Image.fromarray(numpy.uint8(app_zca_img)).show()
