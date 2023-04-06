from re import S
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import torch
from dataloader.data_source import DigitsDataset
from torch.nn import functional as F
from scipy.io import mmread


def read_mtx(filename, dtype='int32'):
    x = mmread(filename).astype(dtype)
    return x

def multi_one_hot(index_tensor, depth_list):
    one_hot_tensor = F.one_hot(index_tensor[:,0], num_classes=depth_list[0])
    for col in range(1, len(depth_list)):
        next_one_hot = F.one_hot(index_tensor[:,col], num_classes=depth_list[col])
        one_hot_tensor = torch.cat([one_hot_tensor, next_one_hot], 1)

    return one_hot_tensor


class SingleCellDataset(DigitsDataset):
    
    def __init__(self, data_name="SingleCell", batch_class=1, raw_data:np.ndarray=None, raw_label:np.ndarray=None, label_batch:np.ndarray=None):
        self.batch_class = batch_class

        data = raw_data.astype(np.float32)
        label = raw_label.astype(np.object_)
        label_batch = label_batch

        # mtx = './data/cd14_monocyte_erythrocyte.mtx'
        # data = read_mtx(mtx)
        # data = data.transpose().todense()
        # label = pd.read_csv('data/cd14_monocyte_erythrocyte_celltype.tsv', sep='\t', header=None).values

        # preprocess
        sadata = anndata.AnnData(X=data)
        sc.pp.normalize_per_cell(sadata, counts_per_cell_after=1e4)
        sadata = sc.pp.log1p(sadata, copy=True)

        if data.shape[1] > 50 and data.shape[0] > 50:
            sc.tl.pca(sadata, n_comps=50)
            data = sadata.obsm['X_pca'].copy()
        else:
            data = sadata.X.copy()

        label_id = list(np.squeeze(label))
        label_id_set = list(set(label_id))
        label_id = np.array([label_id_set.index(i) for i in label_id])
        
        if self.batch_class == [1]:
            batch_hot = np.zeros((data.shape[0], 1)) * -1
            n_batch = self.batch_class
        else:
            n_batch = self.batch_class
            batch_hot = multi_one_hot(torch.tensor(label_batch), n_batch)

        self.data = torch.tensor(data)
        self.label = torch.tensor(label_id)
        self.label_name = [np.array(label)]
        self.batch_hot = torch.tensor(batch_hot).long()

        # sadata.obsm['X_tsne'] = np.load('latent_epoch_299.npy')
        # sc.settings.set_figure_params(dpi=600)
        # sc.pl.tsne(sadata, color='celltype', projection='2d', legend_fontsize='xx-small', save='cd14.png')
