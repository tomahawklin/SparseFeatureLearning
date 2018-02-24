import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter_mul, set_cuda, detach_cuda
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import time

class MultiNet(nn.Module):
    def __init__(self, input_float_dim, embed_dims, embed_sizes, hidden_dim, embed_keys, num_class, lmbda = 0.1):
        super(MultiNet, self).__init__()
        self.linear1 = nn.Linear(input_float_dim, hidden_dim)
        # Embed_keys makes sure that order of layers are consistent
        embed_layers = [nn.Embedding(embed_dims[k], embed_sizes[k]) for k in embed_keys]
        self.embed = nn.ModuleList(embed_layers)
        self.linear2 = nn.Linear(hidden_dim + sum([embed_sizes[k] for k in embed_keys]), hidden_dim)
        sub_hidden_dim = int(hidden_dim / 2)
        self.linear3 = nn.Linear(hidden_dim, sub_hidden_dim)
        self.linear4 = nn.Linear(sub_hidden_dim, 1)
        self.rgrs_loss = nn.L1Loss()
        
    def forward(self, X_float, X_embed, batch_label, batch_ratio):
        float_hid = self.linear1(X_float)
        float_hid = F.relu(float_hid)
        embed_hid = torch.cat([self.embed[i](X_embed[:, i]) for i in range(len(self.embed))], dim = 1)
        hid = torch.cat([float_hid, embed_hid], dim = 1)
        hid = self.linear2(hid)
        hid = F.relu(hid)
        hid = self.linear3(hid)
        hid = F.relu(hid)
        ratio = self.linear4(hid)
        rgrs_loss = self.rgrs_loss(ratio, batch_ratio)
        return rgrs_loss, ratio

def get_stats(ratio_pred, ratio_target):
    ratio_target = detach_cuda(ratio_target).data.numpy().reshape(-1)
    ratio_pred = detach_cuda(ratio_pred).data.numpy().reshape(-1)
    diff = np.abs(ratio_target - ratio_pred)
    return np.mean(diff), np.median(diff), np.max(diff), np.min(diff)

def train(model, train_batches, test_batches, opt, num_epochs, verbose = True):
    epoch = 0
    step = 0
    best_epoch = 0
    best_diff_median = 100
    rpt_step = 5 * num_batch
    test_step = 10 * num_batch
    while epoch < num_epochs:
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ratio = next(train_batches)
        opt.zero_grad()
        loss, ratio = model(batch_X_float, batch_X_embed, batch_label, batch_ratio)
        loss.backward()
        #clip_grad_norm(model.parameters(), 1)
        opt.step()
        step += 1
        if step >= num_batch:
            epoch += 1
            step = 0
        if (step + epoch * num_batch) % rpt_step == 0:
            if verbose:
                mean, median, mmax, mmin = get_stats(ratio, batch_ratio)
                print('Train: step: %d, loss: %.3f, diff mean: %.3f, median: %.3f, max: %.3f, min: %.3f' % ((step + epoch * num_batch), loss.data[0], mean, median, mmax, mmin))
        if (step + epoch * num_batch) % test_step == 0:
            test_X_float, test_X_embed, test_label, test_duration, test_ratio = next(test_batches)
            loss, ratio = model(test_X_float, test_X_embed, test_label, test_ratio)
            if verbose:
                mean, median, mmax, mmin = get_stats(ratio, test_ratio)
                print('Test: step: %d, loss: %.3f, diff mean: %.3f, median: %.3f, max: %.3f, min: %.3f' % ((step + epoch * num_batch), loss.data[0], mean, median, mmax, mmin))
            if median < best_diff_median:
                best_diff_median = median
                best_ratio = ratio
                best_epoch = epoch
                torch.save(model.state_dict(), 'ratio_model.pt')
    return best_diff_median, best_ratio, best_epoch

# Load data
train_data, test_data, embed_dict, embed_dims, embed_keys, float_keys = load_data('final_data.npz')
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

for hidden_dim in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    train_batches = batch_iter_mul(train_data, batch_size, embed_keys, float_keys, shuffle = False)
    test_batches = batch_iter_mul(test_data, len(test_data), embed_keys, float_keys, shuffle = False)
    
    model = MultiNet(len(float_keys), embed_dims, embed_sizes, hidden_dim, embed_keys, num_class)
    model = set_cuda(model)
    opt = optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    start = time.time()
    best_diff, best_ratio, best_epoch = train(model, train_batches, test_batches, opt, num_epochs)
    
    print('Best diff: %.3f in epoch %d. Finished in %.3f seconds' % (best_diff, best_epoch, time.time() - start))


model.load_state_dict(torch.load('ratio_model.pt'))
test_X_float, test_X_embed, test_label, test_duration, test_ratio = next(test_batches)
loss, ratio = model(test_X_float, test_X_embed, test_label, test_ratio)
mean, median, mmax, mmin = get_stats(ratio, test_ratio)



batch_size = 20541
num_batch = int(len(train_data) / batch_size)
train_batches = batch_iter_mul(train_data, batch_size, embed_keys, float_keys, shuffle = False)
train_pred, train_true = [], []
for i in range(num_batch):
    batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ratio = next(train_batches)
    _, train_ratio = model(batch_X_float, batch_X_embed, batch_label, batch_ratio)
    train_pred += train_ratio.data.cpu().numpy().reshape(-1).tolist()
    train_true += batch_ratio.data.cpu().numpy().reshape(-1).tolist()

X_train = np.array(train_pred).reshape(-1, 1)
y_train = np.array([d['label'] for d in train_data])
X_test = ratio.data.cpu().numpy().reshape(-1, 1)
y_test = np.array([d['label'] for d in test_data])
rf_en = RandomForestClassifier(max_depth=10, criterion = 'entropy')
rf_en.fit(X_train, y_train)
pos_prob = rf_en.predict_proba(X_test)[:, 1]
print('AUC: ', roc_auc_score(y_test, pos_prob))


'''
First benchmark: classification with raw features
keys = embed_keys + float_keys
X_train = np.array([[d[k] for k in keys] for d in train_data])
y_train = np.array([d['label'] for d in train_data])
X_test = np.array([[d[k] for k in keys] for d in test_data])
y_test = np.array([d['label'] for d in test_data])
rf_en = RandomForestClassifier(max_depth=10, criterion = 'entropy')
rf_en.fit(X_train, y_train)
pos_prob = rf_en.predict_proba(X_test)[:, 1]
print('AUC: ', roc_auc_score(y_test, pos_prob))
AUC:  0.937743382683

Second benchmark: classification with pymnt_ratio
X_train = np.array([d['pymnt_ratio'] for d in train_data]).reshape(-1, 1)
y_train = np.array([d['label'] for d in train_data])
X_test = np.array([d['pymnt_ratio'] for d in test_data]).reshape(-1, 1)
y_test = np.array([d['label'] for d in test_data])
rf_en = RandomForestClassifier(max_depth=10, criterion = 'entropy')
rf_en.fit(X_train, y_train)
pos_prob = rf_en.predict_proba(X_test)[:, 1]
print('AUC: ', roc_auc_score(y_test, pos_prob))
AUC:  0.981542532219


Notes of best median abs error (median and mean abs error decrease but max abs error increase over time)
L1Loss:
Dim 10:
Best diff: 0.058 in epoch 250.
Dim 20:
Best diff: 0.059 in epoch 230.
Dim 30:
Best diff: 0.057 in epoch 230.
'''