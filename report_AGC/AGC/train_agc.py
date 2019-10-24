from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
from numpy import linalg as la
from heapq import nlargest
from utils import load_data, accuracy, k_means, intra

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--m', type=int, default=7, help='cluster number.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
G, features, labels, idx_train, idx_val, idx_test = load_data()

if args.cuda:
    features = features.cuda()
    G = G.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# def train():
# start_time = time.time()

# convolution, X (n,d)
X = G.mm(features)
# for i in range(10):
# 	X = G.mm(X)
# print(X.shape)
# 转成方阵(n,n)，每个结点的特征之间做一次乘积
K = X.mm(X.T)
# K1 = K.T.mm(K)
# K2 = K.mm(K.T)
# print((K==K.T).all())	# 并非对称矩阵！！
# print(K.equal(K.T))
# print(np.array(K2) == np.array(K1))	# np比较是true，torch比较是false
# print(K2-K1)

W = 0.5*(abs(K)+abs(K.T))
# print(W)

eigenvalue, eigenvector = la.eig(W)
# print(eigenvalue)
# print(eigenvector)		#存在虚数，
eigenvector = np.real(eigenvector)

#找到m个最大的特征值的特征向量，代表整个特征向量
m_largest = list(map(eigenvalue.tolist().index, nlargest(args.m, eigenvalue)))	
# print(m_largest)
# print(eigenvector[:, m_largest])

#k-means
pred_label = k_means(eigenvector[:, m_largest], args.m)
# print(pred_label)
# print(pred_label.max(1)[1].type)
# print(pred_label.sum(0))
print(accuracy(pred_label[idx_train], labels[idx_train]))
intra(pred_label, features)