import torch.nn as nn
import scipy.sparse as sp

class AGC(nn.Module):
    def __init__(self, m):
        super(AGC, self).__init__()
        self.k_means = k_means(m)
        # self.intra = intra(k)

    def forward(self, x, G):
    	# convolution, X (n,d)
		X = G.mm(x)
		
		# 转成方阵(n,n)，保证实对称
		K = X.mm(X.T)
		W = 0.5*(abs(K)+abs(K.T))
		
		# MF
		_, eigenvector = lg.eig(W)
		
		#k-means
		k = self.k_means(eigenvector)

        return x