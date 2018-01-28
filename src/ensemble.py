from utils import load_data, set_cuda, detach_cuda
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

X_train, y_train, X_test, y_test, num_class, input_dim = load_data('e_data.npz')

GBC = GradientBoostingClassifier(n_estimators = 200)
GBC.fit(X_train, y_train)
y_pred = GBC.predict(X_test)
print('accuracy: ',accuracy_score(y_test,y_pred))
target_names = ['Non-Early-Paid','Defaulted','Early-Paid']
print(classification_report(y_test,y_pred,target_names=target_names,digits=4))

rf_en = RandomForestClassifier(max_depth=10, criterion = 'entropy')
rf_en.fit(X_train,y_train)
y_pred = rf_en.predict(X_test)
print('accuracy: ',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=target_names,digits=4))

class EnsembleNet(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_class = 3):
		super(EnsembleNet, self).__init__()
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, num_class)
		self.log_prob = nn.LogSoftmax()
	
	def forward(self, x):
		hid = self.linear1(x)
		hid = F.relu(hid)
		hid = self.linear2(hid)
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
				best_acc = acc
				torch.save(model.state_dict(), 'em_model.pt')
	return best_auc, best_acc

num_samples = X_train.shape[0]
batch_size = 1000
num_batch = int(num_samples / batch_size)
hidden_dim = 4
num_epochs = 50
learning_rate = 1e-2
weight_decay = 5e-5
rpt_step = 10000
test_step = 50000
ratio_0 = sum([1 for t in y_test if t == 0]) / y_train.shape[0]
ratio_1 = sum([1 for t in y_test if t == 1]) / y_train.shape[0]
weight = set_cuda(torch.FloatTensor([1 / ratio_0, 1 / ratio_1, 1 / (1 - ratio_0 - ratio_1)]))
num_class = 3

train_batches = batch_iter(X_train, y_train, batch_size, c1 = GBC, c2 = rf_en, shuffle = True)
test_batches = batch_iter(X_test, y_test, X_test.shape[0], c1 = GBC, c2 = rf_en, shuffle = False)

model = EnsembleNet(2, hidden_dim, num_class)
model = set_cuda(model)
opt = optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
criterion = nn.CrossEntropyLoss()#weight = weight)

best_auc, best_acc = train(model, criterion, train_batches, test_batches, opt, num_epochs)

print('Best auc:%.3f, acc:%.3f' % (best_auc, best_acc))
