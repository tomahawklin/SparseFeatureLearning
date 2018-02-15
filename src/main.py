import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter, set_cuda, detach_cuda, get_stats
import time

class DeepNet(nn.Module):
	def __init__(self, input_float_dim, embed_dims, embed_sizes, hidden_dim, embed_keys, num_class):
		super(DeepNet, self).__init__()
		self.linear1 = nn.Linear(input_float_dim, hidden_dim)
		# Embed_keys makes sure that order of layers are consistent
		embed_layers = [nn.Embedding(embed_dims[k], embed_sizes[k]) for k in embed_keys]
		self.embed = nn.ModuleList(embed_layers)
		self.linear2 = nn.Linear(hidden_dim + sum([embed_sizes[k] for k in embed_keys]), hidden_dim)
		sub_hidden_dim = int(hidden_dim / 2)
		self.linear3 = nn.Linear(hidden_dim, sub_hidden_dim)
		self.linear4 = nn.Linear(sub_hidden_dim, num_class)
		self.log_prob = nn.LogSoftmax()
	
	def forward(self, X_float, X_embed):
		float_hid = self.linear1(X_float)
		float_hid = F.relu(float_hid)
		embed_hid = torch.cat([self.embed[i](X_embed[:, i]) for i in range(len(self.embed))], dim = 1)
		hid = torch.cat([float_hid, embed_hid], dim = 1)
		hid = self.linear2(hid)
		hid = F.relu(hid)
		hid = self.linear3(hid)
		hid = F.relu(hid)
		hid = self.linear4(hid)
		log_prob = self.log_prob(hid)
		return log_prob

def train(model, loss_func, train_batches, test_batches, opt, num_epochs):
	epoch = 0
	step = 0
	best_auc = 0
	best_spec = 0
	best_sens = 0
	best_acc = 0
	best_epoch = 0
	rpt_step = int(num_batch / 5)
	test_step = num_batch
	while epoch < num_epochs:
		batch_X_float, batch_X_embed, batch_y = next(train_batches)
		opt.zero_grad()
		log_prob = model(batch_X_float, batch_X_embed)
		loss = loss_func(log_prob, batch_y)
		loss.backward()
		#clip_grad_norm(model.parameters(), 1)
		opt.step()
		step += 1
		if step >= num_batch:
			epoch += 1
			step = 0
		if (step + epoch * num_batch) % rpt_step == 0:
			_, pred_y = log_prob.data.max(dim = 1)
			if num_class == 2:
				score = log_prob[:, 1]
				acc, auc, sens, spec = get_stats(pred_y, batch_y, score)
				print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f,  sens: %.3f, spec: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc, sens, spec))
			else:
				acc, auc, recall = get_stats(pred_y, batch_y)
				print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc))
				print('Recall:')
				print(recall)
		if (step + epoch * num_batch) % test_step == 0:
			test_X_float, test_X_embed, test_y = next(test_batches)
			log_prob = model(test_X_float, test_X_embed)
			_, test_pred_y = log_prob.data.max(dim = 1)
			if num_class == 2:
				score = log_prob[:, 1]
				acc, auc, sens, spec = get_stats(test_pred_y, test_y, score)
				print('Test: step: %d, acc: %.3f, auc: %.3f, sens: %.3f, spec: %.3f' % ((step + epoch * num_batch), acc, auc, sens, spec))
			else:
				acc, auc, recall = get_stats(test_pred_y, test_batch_y)
				print('Test: step: %d, acc: %.3f, auc: %.3f' % ((step + epoch * num_batch), acc, auc))
				print('Recall:')
				print(recall)
			if auc > best_auc:
				best_epoch = epoch
				best_auc = auc
				best_spec = spec
				best_sens = sens
				best_acc = acc
				torch.save(model.state_dict(), 'deep_model.pt')
	return best_auc, best_acc, best_sens, best_spec, best_epoch

# Load data
train_data, test_data, embed_dict, embed_dims, embed_keys, float_keys = load_data('data_final.npz')

embed_sizes = {'issue_month': 4, 'home_ownership': 2, 'verification_status': 2, 'emp_length': 4, 
               'initial_list_status': 1, 'addr_state': 10, 'early_month': 4, 'grade': 3, 
               'purpose': 5, 'sub_grade': 10, 'zip_code': 20, 'early_year': 6, 'term': 1, 'issue_year': 6}
num_samples = train_data.shape[0]
batch_size = 1000
num_batch = int(num_samples / batch_size)
hidden_dim = 30
num_epochs = 300
learning_rate = 1e-2
weight_decay = 5e-5
ratio_1 = sum([1 for t in train_data if t['label'] == 1]) / train_data.shape[0]
ratio_0 = 1 - ratio_1
weight = set_cuda(torch.FloatTensor([1 / ratio_0, 1 / ratio_1]))
num_class = len(set([t['label'] for t in train_data]))

for hidden_dim in [10, 20, 30, 40, 50, 60]:
    train_batches = batch_iter(train_data, batch_size, embed_keys, float_keys, shuffle = False)
    test_batches = batch_iter(test_data, len(test_data), embed_keys, float_keys, shuffle = False)
    
    model = DeepNet(len(float_keys), embed_dims, embed_sizes, hidden_dim, embed_keys, num_class)
    model = set_cuda(model)
    opt = optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    # TODO: experiment with different weights and dorpout
    criterion = nn.CrossEntropyLoss()#weight = weight)
    
    start = time.time()
    best_auc, best_acc, best_sens, best_spec, best_epoch = train(model, criterion, train_batches, test_batches, opt, num_epochs)
    
    print('Best auc:%.3f, acc:%.3f in epoch %d. Finished in %.3f seconds' % (best_auc, best_acc, best_epoch, time.time() - start))

model.load_state_dict(torch.load('deep_model.pt'))
test_X_float, test_X_embed, test_y = next(test_batches)
log_prob = model(test_X_float, test_X_embed)
_, test_pred_y = log_prob.data.max(dim = 1)
score = log_prob[:, 1]
acc, auc, sens, spec = get_stats(test_pred_y, test_y, score)





'''
With 1 / ratio weights, 100 epochs, 10000 batch_size, 1e-2 lr:
Best auc:0.835, acc:0.831
With 1 / ratio weights, 50 epochs, 10000 batch_size, 1e-2 lr:
Best auc:0.742, acc:0.796
Equal weights, 100 epochs, 10000 batch_size, 1e-2 lr:
Best auc:0.889, acc:0.862
Equal weights, 100 epochs, 1000 batch_size, 1e-2 lr:
Best auc:0.937, acc:0.896
with 1 / ratio weights, 1000 batch_size, 1e-2 lr:
Best auc:0.933, acc:0.892
with 1 / ratio weights, 500 batch_size, 1e-2 lr:
Best auc: 0.940, acc:0.890
with 1 / ratio weights, 1000 batch_size, 5e-3 lr:
Best auc: 0.939, acc:0.882
Equal weights, 100 epochs, 1000 btach_size, 5e-3 lr:
Best auc:0.927, acc:0.890
Equal weights, 300 epochs, 1000 batch_size, 5e-3 lr:
Best auc:0.937, acc:0.895
Equal weights, 300 epochs, 1000 batch_size, 5e-3 lr, 60 hid_dim:
Best auc:0.934, acc:0.891
Equal weights, 300 epochs, 1000 batch_size, 1e-2 lr, 30 hid_dim
Best auc:0.940, acc:0.896
Equal weights, 300 epochs, 1000 batch_size, 1e-2 lr, customize embedding size:
Best auc:0.938, acc:0.895
hidden dim 20:
Best auc:0.945, acc:0.899 customize embedding size
'''