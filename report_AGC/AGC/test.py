import numpy as np
import scipy.sparse as sp
import torch
import time

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def encode_onehot(labels):
    classes = set(labels)   # set无序无索引无重复
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  #{label_class: onehot}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)   #数据中每个文本通过label_class与onehot联系
    # print(classes_dict)
    # print(labels)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_degree(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    d1_2 = sp.diags(r_inv)
    return d1_2

'''
cora 整个语料库中有2708篇论文，在词干堵塞和去除词尾后，只剩下1433个独特的单词，文档频率小于10的所有单词都被删除。
共7类
.content: <paper_id> <word_attributes>+ <class_label>
.cites: <被引论文编号> <引论文编号>
'''

path = "data/cora/"
dataset = "cora"

idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  #读取文件，id，词属性，label
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)    #稀疏矩阵(2708, 1433)  去掉头部id和尾部label
labels = encode_onehot(idx_features_labels[:, -1])     #标签矩阵(2708, 7)

# print(idx_features_labels[:,-1])
# print(features)
# features = normalize(features)
# print(torch.FloatTensor(np.array(features.todense())))
# print(np.where(labels))
# labels = torch.LongTensor(np.where(labels)[1])
# print(labels)
# print(labels.shape)

# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)   #获得头部id
idx_map = {j: i for i, j in enumerate(idx)}                 #重新给每个结点编号
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)     #读取文件，两点之间是否有边 <id,id>
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)  #点id 对应 点编号 <编号,编号>
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)  #稀疏邻接矩阵

# print(idx_features_labels[:, 0])
# print(edges_unordered.shape)
# print(edges)
# print(idx_map.get(35))
# print(idx_map.get(1033))
# print(idx_map.get(954315))
# print(map(idx_map.get, edges_unordered.flatten()))
# print(list(map(idx_map.get, edges_unordered.flatten())))

# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)     #按编号小到大排序
# print(adj)
# adj = normalize(adj + sp.eye(adj.shape[0]))
# adj = sparse_mx_to_torch_sparse_tensor(adj)
# print(adj)
# features = normalize(features)
d1_2 = normalize_degree(adj)
# print(d1_2)

# start_time = time.time()
# I = sp.diags(np.array([1]*adj.shape[0]))
# print(I)
# print('{:.10f}'.format(time.time()-start_time))

features = normalize(features)
features = torch.FloatTensor(np.array(features.todense()))
print(features)
print(features.shape)
