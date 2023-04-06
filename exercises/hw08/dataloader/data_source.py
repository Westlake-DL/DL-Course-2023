from threading import enumerate
import joblib
import torch
from sklearn.metrics import pairwise_distances
import numpy as np
import scipy
from pynndescent import NNDescent
import os
import manifolds.hyperboloid as hyperboloid
import manifolds.poincare as poincare
from manifolds.hyperbolic_project import ToEuclidean, ToSphere, ToPoincare, ToLorentz


class DigitsDataset(torch.utils.data.Dataset):
    
    def __init__(self, **kwargs):
        self.args = kwargs
    
    def _Pretreatment(self, data, metric_s='euclidean', K=5):

        self.rie_pro_input = ToEuclidean()
        rho, sigma = self._initKNN(self.rie_pro_input(data), metric_s=metric_s, K=K)

        self.sigma = sigma
        self.rho = rho

    def _initKNN(self, X, metric_s='euclidean', K=5):
        
        print('use kNN method to find the sigma')

        X_rshaped = X.reshape((X.shape[0],-1))
        index = NNDescent(X_rshaped, n_jobs=-1, metric=metric_s)
        self.neighbors_index, neighbors_dist = index.query(X_rshaped, k=K)
        neighbors_dist = np.power(neighbors_dist, 2)

        rho = np.zeros(neighbors_dist.shape[0])
        sigma = np.ones(neighbors_dist.shape[0])
        
        return rho, sigma
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        batch_item = self.batch_hot[index]
            
        return batch_item, index
