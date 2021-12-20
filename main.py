import os.path
import numpy as np
from DataLoader import *
from Solver import *
import torch
import torch.nn as nn
from model import *
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

dataset = '.\dataset\cifar-100-python'

file = os.path.join(dataset, 'train')
with open(file, 'rb') as f:
    train_set = load_pickle(f)
tr_X, tr_Y = load_dataset(file)
print('tr_X: ', tr_X.shape)
print('tr_Y: ', tr_Y.shape)

file = os.path.join(dataset, 'test')
with open(file, 'rb') as f:
    test_set = load_pickle(f)
te_X, te_Y = load_dataset(file)
print('te_X: ', te_X.shape)
print('te_Y: ', te_Y.shape)

tr_X = torch.from_numpy(tr_X).to(torch.float32)
tr_Y = torch.from_numpy(tr_Y).to(torch.int64)
te_X = torch.from_numpy(te_X).to(torch.float32)
te_Y = torch.from_numpy(te_Y).to(torch.int64)

small_data = 1000
small_tr_X = tr_X[:small_data]
small_tr_Y = tr_Y[:small_data]

model1 = nn.Sequential(
    nn.Conv2d(3, 128, 5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(128, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    Flatten(),
    nn.Linear(64 * 8 * 8, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 100)
)
model1.to(device)

# train1
his_test_self_1 = []
his_test_pre_1 = []
for i in range(10):
    class_start = 10 * i
    class_end = 10 * (i + 1)
    tmp_tr_X = tr_X[(tr_Y < class_end) & (tr_Y >= class_start)]
    tmp_tr_Y = tr_Y[(tr_Y < class_end) & (tr_Y >= class_start)]
    tmp_te_self_X = te_X[(te_Y < class_end) & (te_Y >= class_start)]
    tmp_te_self_Y = te_Y[(te_Y < class_end) & (te_Y >= class_start)]
    tmp_te_X = te_X[te_Y < class_end]
    tmp_te_Y = te_Y[te_Y < class_end]
    solver = Solver(
        model1,
        {
            'tr_X': tmp_tr_X,
            'tr_Y': tmp_tr_Y,
            'val_X': tmp_te_X,
            'val_Y': tmp_te_Y
        },
        num_epochs=10,
        print_every=10,
        verbose=True,
        batch_size=200
    )
    print('0 to {}: '.format(class_end - 1))
    print('tmp_tr_X: ', tmp_tr_X.shape)
    print('tmp_tr_Y: ', tmp_tr_Y.shape)
    print('tmp_te_X: ', tmp_te_X.shape)
    print('tmp_te_Y: ', tmp_te_Y.shape)
    solver.train()
    pre_acc = solver.check_accuracy(tmp_te_X, tmp_te_Y)
    self_acc = solver.check_accuracy(tmp_te_self_X, tmp_te_self_Y)
    his_test_pre_1.append(pre_acc)
    his_test_self_1.append(self_acc)
    print('test_acc: ', pre_acc)

model2 = Model()
model2.to(device)

# 0.2619
# train 2
his_test_self_2 = []
his_test_pre_2 = []
for i in range(10):
    if i > 0:
        model2.add_node(10)
    class_start = 10 * i
    class_end = 10 * (i + 1)
    tmp_tr_X = tr_X[(tr_Y < class_end) & (tr_Y >= class_start)]
    tmp_tr_Y = tr_Y[(tr_Y < class_end) & (tr_Y >= class_start)]
    tmp_te_self_X = te_X[(te_Y < class_end) & (te_Y >= class_start)]
    tmp_te_self_Y = te_Y[(te_Y < class_end) & (te_Y >= class_start)]
    tmp_te_X = te_X[te_Y < class_end]
    tmp_te_Y = te_Y[te_Y < class_end]
    solver = Solver(
        model2,
        {
            'tr_X': tmp_tr_X,
            'tr_Y': tmp_tr_Y,
            'val_X': tmp_te_X,
            'val_Y': tmp_te_Y
        },
        num_epochs=10,
        print_every=10,
        verbose=True,
        batch_size=200
    )
    print('0 to {}: '.format(class_end - 1))
    print('tmp_tr_X: ', tmp_tr_X.shape)
    print('tmp_tr_Y: ', tmp_tr_Y.shape)
    print('tmp_te_X: ', tmp_te_X.shape)
    print('tmp_te_Y: ', tmp_te_Y.shape)
    solver.train()
    pre_acc = solver.check_accuracy(tmp_te_X, tmp_te_Y)
    self_acc = solver.check_accuracy(tmp_te_self_X, tmp_te_self_Y)
    his_test_pre_2.append(pre_acc)
    his_test_self_2.append(self_acc)
    print('test_acc: ', pre_acc)

a = np.arange(10, 101, 10)
plt.subplot(2, 1, 1)
plt.title("pre_classes_acc")
plt.xlabel("num_classes")
plt.ylabel("acc")
plt.xticks(np.linspace(0, 100, 11))
plt.plot(a, his_test_pre_1, 'r')
plt.plot(a, his_test_pre_2, 'b')

plt.subplot(2, 1, 2)
plt.title("self_classes_acc")
plt.xlabel('num_classes')
plt.ylabel('acc')
plt.xticks(np.linspace(0, 100, 11))
plt.plot(a, his_test_self_1, 'r')
plt.plot(a, his_test_self_2, 'b')
plt.show()