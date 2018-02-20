import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter_mul, set_cuda, detach_cuda, get_stats
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
        self.linear4_1 = nn.Linear(sub_hidden_dim, 1)
        self.log_prob = nn.LogSoftmax()
        self.linear4_2 = nn.Linear(sub_hidden_dim + 1, num_class)
        self.lmbda = lmbda
        self.rgrs_loss = nn.MSELoss()
        self.clas_loss = nn.CrossEntropyLoss()
    
    def forward(self, X_float, X_embed, batch_label, batch_ratio):
        float_hid = self.linear1(X_float)
        float_hid = F.relu(float_hid)
        embed_hid = torch.cat([self.embed[i](X_embed[:, i]) for i in range(len(self.embed))], dim = 1)
        hid = torch.cat([float_hid, embed_hid], dim = 1)
        hid = self.linear2(hid)
        hid = F.relu(hid)
        hid = self.linear3(hid)
        hid = F.relu(hid)
        ratio = self.linear4_1(hid)
        rgrs_loss = self.rgrs_loss(ratio, batch_ratio)
        diff = torch.mean(ratio) - torch.mean(batch_ratio)
        clas_input = torch.cat([hid, ratio], dim = 1)
        score = self.linear4_2(clas_input)
        log_prob = self.log_prob(score)
        clas_loss = self.clas_loss(log_prob, batch_label)
        return clas_loss + self.lmbda * rgrs_loss, log_prob, diff.data[0]

def train(model, train_batches, test_batches, opt, num_epochs, verbose = True):
    epoch = 0
    step = 0
    best_auc = 0
    best_spec = 0
    best_sens = 0
    best_acc = 0
    best_epoch = 0
    best_diff = 0
    rpt_step = 10 * num_batch
    test_step = 20 * num_batch
    while epoch < num_epochs:
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ratio = next(train_batches)
        opt.zero_grad()
        loss, log_prob, diff = model(batch_X_float, batch_X_embed, batch_label, batch_ratio)
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
                acc, auc, sens, spec = get_stats(pred_y, batch_label, score)
                if verbose:
                    print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f,  sens: %.3f, spec: %.3f, diff: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc, sens, spec, diff))
            else:
                acc, auc, recall = get_stats(pred_y, batch_label)
                print('Train: step: %d, avg loss: %.3f, acc: %.3f, auc: %.3f' % ((step + epoch * num_batch), loss.data[0], acc, auc))
                print('Recall:')
                print(recall)
        if (step + epoch * num_batch) % test_step == 0:
            test_X_float, test_X_embed, test_label, test_duration, test_ratio = next(test_batches)
            _, log_prob, diff = model(test_X_float, test_X_embed, test_label, test_ratio)
            _, test_pred_y = log_prob.data.max(dim = 1)
            if num_class == 2:
                score = log_prob[:, 1]
                acc, auc, sens, spec = get_stats(test_pred_y, test_label, score)
                if verbose:
                    print('Test: step: %d, acc: %.3f, auc: %.3f, sens: %.3f, spec: %.3f, diff: %.3f' % ((step + epoch * num_batch), acc, auc, sens, spec, diff))
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
                best_diff = diff
                torch.save(model.state_dict(), 'multi_model.pt')
    return best_auc, best_acc, best_sens, best_spec, best_epoch, best_diff

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
    best_auc, best_acc, best_sens, best_spec, best_epoch, best_diff = train(model, train_batches, test_batches, opt, num_epochs)
    
    print('Best auc:%.3f, acc:%.3f, diff: %.3f in epoch %d. Finished in %.3f seconds' % (best_auc, best_acc, best_diff, best_epoch, time.time() - start))

model.load_state_dict(torch.load('multi_model.pt'))
test_X_float, test_X_embed, test_label, test_duration, test_ratio = next(test_batches)
_, log_prob, diff = model(test_X_float, test_X_embed, test_label, test_duration)
_, test_pred_y = log_prob.data.max(dim = 1)
score = log_prob[:, 1]
acc, auc, sens, spec = get_stats(test_pred_y, test_label, score)



'''
Multi-task learning notes
Dim 20, regression and classification in paralell
Best auc:0.714, acc:0.754, diff: -0.037 in epoch 300
Dim 30:
Best auc:0.722, acc:0.758, diff: -0.025 in epoch 300
Dim 40:
Best auc:0.723, acc:0.757, diff: -0.029 in epoch 300
Dim 50:
Best auc:0.724, acc:0.758, diff: -0.024 in epoch 300

Dim 20, sequentail regression (duration) + classification
Best auc:0.716, acc:0.755, diff: -0.050 in epoch 300 

Dim20,  sequentail regression (ratio) + classification
Best auc:0.729, acc:0.760, diff: -0.001 in epoch 190
'''