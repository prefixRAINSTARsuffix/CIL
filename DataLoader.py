import platform

import numpy as np
import pickle


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    if version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError('invalid python version: {}'.format(version))


# 返回  ndarray 类型
def load_dataset(file):
    with open(file, 'rb') as f:
        data = load_pickle(f)
        trX = data['data'].reshape(-1, 3, 32, 32)
        trY = np.array(data['fine_labels']).reshape(-1)
    return trX, trY


# 按 label 分类
# (max_label, )
def load_by_label(X, Y):
    max_label = max(Y)
