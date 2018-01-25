import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm

class DeepNet(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_class = 2):
		super(DeepNet, self).__init__()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
		self.linear3 = nn.Linear(int(hidden_dim / 2), num_class)
		self.log_prob = nn.LogSoftmax()
	
	def forward(self, x):
		hid = self.linear1(x)
		hid = F.relu(hid)
		hid = self.linear2(hid)
		hid = F.relu(hid)
		hid = self.linear3(hid)
		log_prob = self.log_prob(hid)
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
	sens, spec = precision_recall_fscore_support(detach_cuda(batch_y).data.numpy(), detach_cuda(pred_y).numpy())[1]
	return acc, auc, sens, spec

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
			acc, auc, sens, spec = get_stats(pred_y, batch_y)
			#print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f,  sens: %.3f, spec: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc, sens, spec))
		if step % test_step == 0:
			test_batch_X, test_batch_y = next(test_batches)
			log_prob = model(test_batch_X)
			_, test_pred_y = log_prob.data.max(dim = 1)
			acc, auc, sens, spec = get_stats(test_pred_y, test_batch_y)
			#print('Test: step: %d, acc: %.3f, auc: %.3f, sens: %.3f, spec: %.3f' % ((step + epoch * num_batch), acc, auc, sens, spec))
			if auc > best_auc:
				best_auc = auc
				best_spec = spec
				best_sens = sens
				best_acc = acc
				torch.save(model.state_dict(), 'model.pt')
	return best_auc, best_acc, best_sens, best_spec

# Load data
X_train, y_train, X_test, y_test, num_class, input_dim = load_data('b_data.npz')

num_samples = X_train.shape[0]
batch_size = 1000
num_batch = int(num_samples / batch_size)
hidden_dim = 30
num_epochs = 50
learning_rate = 1e-2
weight_decay = 5e-5
rpt_step = 10000
test_step = 50000
ratio = sum(y_train) / y_train.shape[0] / 10
weight = set_cuda(torch.FloatTensor([1 / (1 - ratio), 1 / ratio]))

train_batches = batch_iter(X_train, y_train, batch_size, shuffle = True)
test_batches = batch_iter(X_test, y_test, X_test.shape[0], shuffle = False)

model = DeepNet(input_dim, hidden_dim)
model = set_cuda(model)
opt = optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
criterion = nn.CrossEntropyLoss(weight = weight)

best_auc, best_acc, best_sens, best_spec = train(model, criterion, train_batches, test_batches, opt, num_epochs)

print('Best auc:%.3f, acc:%.3f, sens:%.3f, spec: %.3f' % (best_auc, best_acc, best_sens, best_spec))

#model.load_state_dict(torch.load('model.pt'))




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
'''
from sklearn.ensemble import RandomForestClassifier
rf_Gini = RandomForestClassifier(max_depth=10)
rf_Gini.fit(X_train,y_train)
y_pred = rf_Gini.predict(X_test)
print('accuracy: ',accuracy_score(y_test,y_pred))
target_names = ['Non-Defaulted Loan','Defaulted Loan']
print(classification_report(y_test,y_pred,target_names=target_names,digits=4))
print('AUC: ',roc_auc_score(y_test,y_pred))
'''
