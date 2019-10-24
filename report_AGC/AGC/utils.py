import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))  # D^-1 A 
    adj = normalize_degree(adj + sp.eye(adj.shape[0]))  # D^-1/2 A D^-1/2
    I = sp.diags(np.array([1]*adj.shape[0]))
    G = 0.5 * (I + adj)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])   
    # adj = sparse_mx_to_torch_sparse_tensor(adj)   
    G = sparse_mx_to_torch_sparse_tensor(G)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return G, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# D^-1/2 A D^-1/2
def normalize_degree(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    degree = sp.diags(r_inv)
    return degree.dot(mx).dot(degree)

def accuracy(output, labels):
    print(output.max(1)[1][:10])
    print(labels[:10])
    preds = output.max(1)[1].type_as(labels)
    
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def k_means(eigenvector, m):
    kmeans = KMeans(n_clusters=m, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(eigenvector)
    pred_label = sp.coo_matrix((np.ones(eigenvector.shape[0]), (range(len(pred_y)), pred_y)), shape=(len(pred_y), m), dtype=np.int32)
    return torch.LongTensor(np.array(pred_label.todense()))

def intra(pred_label, features):
    C = pred_label.sum(0)   # 聚成7类，各类的总数 (1,7)
    C_sum = sum(C)
    C_dict = {}
    for i in range(pred_label.shape[1]):
        u = pred_label[:,i]     # 获得每一列
        l = []
        for j in range(u.shape[0]):
            if u[j] == 1:       
                l.append(j)
        C_dict[i] = l           # 将在同一组的结点归到一个list
    # print(C_dict)
    
    C_dict_sum = []
    for i in range(pred_label.shape[1]):
        dict_sum = 0
        for j in C_dict[i]:
            for k in C_dict[i]:
                if j != k:
                    dict_sum += sum((features[j] - features[k])**2)
        C_dict_sum.append(dict_sum)
    print(C_dict_sum)

def d_intra(k):
    pass