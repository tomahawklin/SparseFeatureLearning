import numpy as np
import random
import torch
from torch.autograd import Variable

def load_data(path):
    data_file = np.load(path)
    train = data_file['train']
    test = data_file['test']
    embed_dict = data_file['feature_dict'].item()
    embed_dims = {k: len(embed_dict[k]) for k in embed_dict}
    embed_keys = list(embed_dict.keys())
    float_keys = [k for k in train[0] if k not in embed_keys]
    float_keys.remove('label')
    return train, test, embed_dict, embed_dims, embed_keys, float_keys

def struct_data(data_dic, embed_keys, float_keys):
	y = int(data_dic['label'])
	X_float = [data_dic[k] for k in float_keys] 
	X_embed = [int(data_dic[k]) for k in embed_keys]
	return X_float, X_embed, y

def batch_iter(data, batch_size, embed_keys, float_keys, shuffle = False):
	start = -1 * batch_size
	num_samples = len(data)
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
		batch_data = data[batch_idx]
		batch_X_float, batch_X_embed, batch_y = [], [], [] 
		for i in range(len(batch_data)):
			X_float, X_embed, y = struct_data(train[0], embed_keys, float_keys)
			batch_y.append(y)
			batch_X_float.append(X_float)
			batch_X_embed.append(X_embed)
		batch_X_float = Variable(torch.FloatTensor(batch_X_float))
		batch_X_embed = Variable(torch.LongTensor(batch_X_embed))
		batch_y = Variable(torch.LongTensor(batch_y))
		yield set_cuda(batch_X_float), set_cuda(batch_X_embed), set_cuda(batch_y)

def batch_iter_ensemble(X, y, batch_size, c1 = None, c2 = None, shuffle = False):
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
