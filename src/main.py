import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter, set_cuda, detach_cuda

class DeepNet(nn.Module):
	def __init__(self, input_float_dim, input_embed_dim, embed_size, hidden_dim, embed_keys, num_class = 2):
		super(DeepNet, self).__init__()
		self.float = nn.Linear(input_float_dim, hidden_dim)
		# Note: cuda will not move list structure to gpu. Explicitly define layers?
		self.embed = [nn.Embedding(input_embed_dim[k], embed_size[k]) for k in embed_keys]
		self.linear2 = nn.Linear(hidden_dim + sum([embed_size[k] for k in embed_keys]), int(hidden_dim / 2))
		self.linear3 = nn.Linear(int(hidden_dim / 2), num_class)
		self.log_prob = nn.LogSoftmax()
	# TODO
	def forward(self, X_float, X_embed):
		float_hid = self.float(X_float)
		float_hid = F.relu(float_hid)
		hid = self.linear2(hid)
		hid = F.relu(hid)
		hid = self.linear3(hid)
		log_prob = self.log_prob(hid)
		return log_prob

def get_stats(pred_y, batch_y):
	# Todo: add recall, etc
	total = batch_y.size()[0]
	correct = torch.sum(pred_y == batch_y.data)
	acc = float(correct / total)
	batch_y = detach_cuda(batch_y).data.numpy()
	pred_y = detach_cuda(pred_y).numpy()
	try:
		auc = roc_auc_score(batch_y, pred_y)
		sens, spec = precision_recall_fscore_support(batch_y, pred_y)[1]
		return acc, auc, sens, spec
	except:
		auc = float('nan')
		recall = precision_recall_fscore_support(batch_y, pred_y)[1]
	return acc, auc, recall

def train(model, loss_func, train_batches, test_batches, opt, num_epochs):
	epoch = 0
	step = 0
	best_auc = 0
	best_spec = 0
	best_sens = 0
	best_acc = 0
	while epoch < num_epochs:
		batch_X, batch_y = next(train_batches)
		opt.zero_grad()
		log_prob = model(batch_X)
		loss = loss_func(log_prob, batch_y)
		loss.backward()
		#clip_grad_norm(model.parameters(), 1)
		opt.step()
		step += 1
		if step >= num_batch:
			epoch += 1
			step = 0
		if step % rpt_step == 0:
			_, pred_y = log_prob.data.max(dim = 1)
			if num_class == 2:
				acc, auc, sens, spec = get_stats(pred_y, batch_y)
				#print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f,  sens: %.3f, spec: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc, sens, spec))
			else:
				acc, auc, recall = get_stats(pred_y, batch_y)
				print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc))
				print('Recall:')
				print(recall)
		if step % test_step == 0:
			test_batch_X, test_batch_y = next(test_batches)
			log_prob = model(test_batch_X)
			_, test_pred_y = log_prob.data.max(dim = 1)
			if num_class == 2:
				acc, auc, sens, spec = get_stats(test_pred_y, test_batch_y)
				#print('Test: step: %d, acc: %.3f, auc: %.3f, sens: %.3f, spec: %.3f' % ((step + epoch * num_batch), acc, auc, sens, spec))
			else:
				acc, auc, recall = get_stats(test_pred_y, test_batch_y)
				print('Test: step: %d, acc: %.3f, auc: %.3f' % ((step + epoch * num_batch), acc, auc))
				print('Recall:')
				print(recall)
			if acc > best_acc:
				best_auc = auc
				#best_spec = spec
				#best_sens = sens
				best_acc = acc
				torch.save(model.state_dict(), 'e_model.pt')
	return best_auc, best_acc#, best_sens, best_spec

# Load data
train, test, embed_dict, embed_dims, embed_keys, float_keys = load_data('data_final.npz')

num_samples = X_train.shape[0]
batch_size = 1000
num_batch = int(num_samples / batch_size)
hidden_dim = 30
num_epochs = 100
learning_rate = 1e-2
weight_decay = 5e-5
rpt_step = 10000
test_step = 50000
ratio_0 = sum([1 for t in y_test if t == 0]) / y_train.shape[0]
ratio_1 = sum([1 for t in y_test if t == 1]) / y_train.shape[0]
weight = set_cuda(torch.FloatTensor([1 / ratio_0, 1 / ratio_1, 1 / (1 - ratio_0 - ratio_1)]))
num_class = 3

train_iter = batch_iter(train, batch_size, embed_keys, float_keys, shuffle = False)
test_iter = batch_iter(test, 1000, embed_keys, float_keys, shuffle = False)

model = DeepNet(input_dim, hidden_dim, num_class)
model = set_cuda(model)
opt = optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
criterion = nn.CrossEntropyLoss()#weight = weight)

best_auc, best_acc = train(model, criterion, train_batches, test_batches, opt, num_epochs)

print('Best auc:%.3f, acc:%.3f' % (best_auc, best_acc))

#model.load_state_dict(torch.load('model.pt'))



