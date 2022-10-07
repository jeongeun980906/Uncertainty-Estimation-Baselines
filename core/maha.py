import torch
import numpy as np
from sklearn.cluster import KMeans 

class maha_distance():
    def __init__(self,type,num_cluster):
        self.num_cluster = num_cluster
        self.type = type
        self.T = 1000

    def model_feature(self,ftrain):
        ftrain = np.asarray(ftrain)
        self.mu_0 = np.mean(ftrain,axis=0)
        self.sigma_0 = np.cov(ftrain.T)
        print(self.sigma_0.shape)
        kmeans = KMeans(n_clusters=self.num_cluster)
        kmeans.fit(ftrain)
        mu_k = []
        std = np.eye(ftrain.shape[1])
        size = ftrain.shape[0]
        label = kmeans.labels_
        for i in range(self.num_cluster):
            index = np.where(label == i)[0]
            data = ftrain[index]
            mu = np.mean(data,axis=0)
            a = np.expand_dims((data-mu),-1)
            pa = np.transpose(a,(0,2,1))
            # print(a.shape,pa.shape)
            temp = np.matmul(a,pa) # [M x D x D]
            temp = np.sum(temp,axis=0)
            mu_k.append(mu)
            std += temp 
        self.std_k = std/size # [D x D]
        self.mu_k = np.vstack(mu_k) # [K x D]

    def score(self,ftest):
        '''
        ftest [N x D] 
        mu_k [K x D]
        std_k [D x D]
        (ftest - mu_k) std_k^-1 (ftet-mu_k)
        '''
        std = self.std_k
        maha_total = []
        print(self.type)
        for i in range(self.num_cluster):
            mu = self.mu_k[i]
            dis = ftest-mu # [N x D]
            dis = np.expand_dims(dis,axis=1) # [N x 1 x D]
            maha = np.matmul(dis,std) 
            maha = np.matmul(maha,np.transpose(dis,(0,2,1)))[:,0,0] # [N x 1 x 1]
            maha_total.append(maha)
        maha_total = np.vstack(maha_total)
        if self.type == 'MD':
            return np.min(maha_total,axis=0)
        elif self.type == 'RMD':
            dis = ftest-self.mu_0 # [N x D]
            dis = np.expand_dims(dis,axis=1) # [N x 1 x D]
            maha_0 = np.matmul(dis,self.sigma_0) 
            maha_0 = np.matmul(maha_0,np.transpose(dis,(0,2,1)))[:,0,0] # [N x 1 x 1]
            return np.min(maha_total,axis=0) - maha_0
        elif self.type == 'DMD':
            maha = np.min(maha_total,axis=0)
            print(np.mean(maha_total))
            p_c = np.exp(-maha_total/self.T) / np.sum(np.exp(-maha_total/self.T),axis=0)
            # h_c = -p_c*np.log(p_c)
            h_c = -np.sum(p_c*np.log(p_c),axis=0)
            dis = ftest-self.mu_0 # [N x D]
            dis = np.expand_dims(dis,axis=1) # [N x 1 x D]
            maha_0 = np.matmul(dis,self.sigma_0) 
            maha_0 = np.matmul(maha_0,np.transpose(dis,(0,2,1)))[:,0,0] # [N x 1 x 1]
            p_o = np.exp(-maha_0/self.T) / np.sum(np.exp(-maha_total/self.T),axis=0)
            # h_o = p_o*np.log(p_o)
            return h_c #-h_o