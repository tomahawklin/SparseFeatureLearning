import numpy as np
import random
import torch
from torch.autograd import Variable

def load_data(path):
    data = np.load(path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    num_class = np.unique(y_train).shape[0]
    input_dim = X_train.shape[1]
    return X_train, y_train, X_test, y_test, num_class, input_dim

def batch_iter(X, y, batch_size, c1 = None, c2 = None, shuffle = False):
	start = -1 * batch_size
	num_samples = X.shape[0]
	indices = list(range(num_samples))
	if shuffle:
		random.shuffle(indices)
	while True:
		start += batch_size
		if start >= num_samples - batch_size:
			if shuffle:
				random.shuffle(indices)
			batch_idx = indices[:batch_size]
			start = batch_size
		else:
			batch_idx = indices[start: start + batch_size]
		batch_X = X[batch_idx]
		if c1 is not None and c2 is not None:
			c1_X = c1.predict(batch_X)
			c2_X = c2.predict(batch_X)
			batch_X = [c1_X.tolist(), c2_X.tolist()]
			batch_X = Variable(torch.FloatTensor(batch_X).transpose(0, 1))
		else:
			batch_X = Variable(torch.FloatTensor(batch_X))
		batch_y = y[batch_idx]
		batch_y = Variable(torch.LongTensor(batch_y))
		yield set_cuda(batch_X), set_cuda(batch_y)


def set_cuda(var):
	if torch.cuda.is_available():
		return var.cuda()
	else:
		return var

def detach_cuda(var):
	if torch.cuda.is_available():
		return var.cpu()
	else:
		return var
