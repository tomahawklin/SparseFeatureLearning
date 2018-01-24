import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class DeepNet(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_class = 2):
		super(DeepNet, self).__init__()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, num_class)
		self.log_prob = nn.LogSoftmax()
	
	def forward(self, x):
		hid = self.linear1(x)
		hid = F.relu(hid)
		out = self.linear2(hid)
		log_prob = self.log_prob(out)
		return log_prob

def load_data(path):
    data = np.load(path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    num_class = np.unique(y_train).shape[0]
    input_dim = X_train.shape[1]
    return X_train, y_train, X_test, y_test, num_class, input_dim

def batch_iter(X, y, batch_size, shuffle = False):
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
		batch_y = y[batch_idx]
		batch_X = Variable(torch.FloatTensor(batch_X))
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

def get_stats(pred_y, batch_y):
	# Todo: add recall, etc
	total = batch_y.size()[0]
	correct = torch.sum(pred_y == batch_y.data)
	acc = float(correct / total)
	auc = roc_auc_score(detach_cuda(batch_y).data.numpy(), detach_cuda(pred_y).numpy())
	return acc, auc

# Load data
X_train, y_train, X_test, y_test, num_class, input_dim = load_data('data.npz')

num_samples = X_train.shape[0]
batch_size = 100
num_batch = num_samples / batch_size
hidden_dim = 30
num_epochs = 100
learning_rate = 0.05
rpt_step = 1000
test_step = 5000

train_batches = batch_iter(X_train, y_train, batch_size, shuffle = True)
test_batches = batch_iter(X_test, y_test, X_test.shape[0], shuffle = False)

model = DeepNet(input_dim, hidden_dim)
model = set_cuda(model)
model.train()
opt = optim.Adagrad(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

epoch = 1
step = 1
while epoch < num_epochs:
	batch_X, batch_y = next(train_batches)
	opt.zero_grad()
	log_prob = model(batch_X)
	loss = criterion(log_prob, batch_y)
	loss.backward()
	opt.step()
	step += 1
	if step >= num_batch:
		epoch += 1
		step = 1
	if step % rpt_step == 0:
		_, pred_y = log_prob.data.max(dim = 1)
		acc, auc = get_stats(pred_y, batch_y)
		print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f' % (step, loss.data[0], acc, auc))
	if step % test_step == 0:
		batch_X, batch_y = next(test_batches)
		model.eval()
		log_prob = model(batch_X)
		_, pred_y = log_prob.data.max(dim = 1)
		acc, auc = get_stats(pred_y, batch_y)
		print('Test: step: %d, acc: %.3f, auc: %.3f' % (step, acc, auc))
		model.train()






'''
# Simple gradient boositing classifier example that reaches comparable results without parameter tunning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)
y_pred = GBC.predict(X_test)
print('accuracy: ',accuracy_score(y_test,y_pred))
target_names = ['Non-Defaulted Loan','Defaulted Loan']
print(classification_report(y_test,y_pred,target_names=target_names,digits=4))
print('AUC: ',roc_auc_score(y_test,y_pred))
'''

