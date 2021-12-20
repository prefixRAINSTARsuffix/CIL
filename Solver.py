import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class Solver(object):
    """
    kwargs:
    optimizer='sgd'
    optim_config={
        'learning_rate': 0.001
        'momentum': 0
        'nesterov': False
    }
    loss='cross_entropy'
    num_epochs=5
    batch_size=200
    print_every=100
    verbose=False

    data: dict {'tr_X', 'tr_Y', 'val_X', 'val_Y'}
    """

    def __init__(self, model, data, **kwargs):
        self.model = model
        # 训练集
        self.tr_X = data['tr_X']
        self.tr_Y = data['tr_Y']
        # 验证集
        self.val_X = data['val_X']
        self.val_Y = data['val_Y']

        self.optimizer = kwargs.pop('optimizer', 'sgd')
        self.loss = kwargs.pop('loss', 'cross_entropy')
        self.optim_config = kwargs.pop('optim_config', {
            'learning_rate': 0.001,
            'momentum': 0,
            'nesterov': False
        })
        self.num_epochs = kwargs.pop('num_epochs', 5)
        self.batch_size = kwargs.pop('batch_size', 200)
        self.print_every = kwargs.pop('print_every', 100)
        # 是否输出 print_every
        self.verbose = kwargs.pop('verbose', False)
        self.loss_history = []
        # 是否支持 cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # loss_function
        if self.loss == 'cross_entropy':
            self.loss = F.cross_entropy
        else:
            self.loss = F.cross_entropy
        #  optimizer
        if self.optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(),
                                       lr=self.optim_config['learning_rate'],
                                       momentum=self.optim_config['momentum'],
                                       nesterov=self.optim_config['nesterov']
                                       )
        else:
            self.optimizer = optim.Adam(model.parameters(),
                                        betas=self.optim_config['betas'],
                                        lr=self.optim_config['learning_rate'])

    # 更新一个 iter
    def step(self, cur_iteration=0):
        num_tr = self.tr_X.shape[0]
        it_start = self.batch_size * cur_iteration
        if it_start >= num_tr:
            raise ValueError('Too large iteration: {}'.format(cur_iteration))
        it_end = min(it_start + self.batch_size, num_tr)
        X_batch = self.tr_X[it_start:it_end].to(self.device)
        Y_batch = self.tr_Y[it_start:it_end].to(self.device)
        Y_pred = self.model(X_batch)

        loss = self.loss(Y_pred, Y_batch)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss)

        # pred <-- argmax(Y_pred)
        pred = torch.argmax(Y_pred, dim=1)
        return torch.sum(pred == Y_batch)

    def check_accuracy(self, X, Y):
        with torch.no_grad():
            num_X = X.shape[0]
            iterations_per_epoch = (num_X + self.batch_size - 1) // self.batch_size
            sum_success = 0
            it_start = 0
            it_end = min(self.batch_size, num_X)
            for iter in range(iterations_per_epoch):
                X_batch = X[it_start:it_end]
                Y_batch = Y[it_start:it_end]
                Y_pred = self.model(X_batch)
                pred = torch.argmax(Y_pred, dim=1)
                sum_success += torch.sum(pred == Y_batch)
                it_start = it_end
                it_end = min(num_X, it_end+self.batch_size)
            return sum_success / num_X

    def train(self):
        num_tr = self.tr_X.shape[0]
        iterations_per_epoch = (num_tr + self.batch_size - 1) // self.batch_size
        tot_iterations = self.num_epochs * iterations_per_epoch
        sum_success = 0
        for iter in range(tot_iterations):
            batch_success = self.step(iter % iterations_per_epoch)
            sum_success += batch_success

            if self.verbose and iter % self.print_every == 0:
                print('Iteration {} / {}  loss: {}'.format(iter, tot_iterations, self.loss_history[-1]))
