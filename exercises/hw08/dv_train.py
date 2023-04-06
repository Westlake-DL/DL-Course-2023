import os
from re import L, X

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from torch.optim.lr_scheduler import StepLR
from matplotlib.projections import register_projection

import Loss.dmt_loss_aug as Loss_use_aug
import nuscheduler

from dataloader import data_base
from manifolds.hyperbolic_project import ToEuclidean, ToPoincare, ToLorentz
import scanpy as sc

torch.set_num_threads(1)


class Generator(nn.Module):
    def __init__(self, dims):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.1))
            return layers

        self.model = nn.Sequential(
            *block(int(np.array(dims)), 500),
            *block(500, 300),
            *block(300, 100),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_latent_dim):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.1))
            return layers

        self.model = nn.Sequential(
            *block(100, 300),
            *block(300, 100),
            nn.Linear(100, num_latent_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DV_Model(LightningModule):
    
    def __init__(
        self,
        **kwargs,
        ):

        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.learning_rate = self.hparams.learning_rate
        self.batch_size = self.hparams.batch_size
        self.num_latent_dim = self.hparams.num_latent_dim
        self.data_name = self.hparams.data_name
        self.batch_class = self.hparams.batch_class
        self.setup()
        
        self.dims = self.data_train.data[0].shape
        self.log_interval = self.hparams.log_interval
        self.c_input = self.hparams.c_input
        self.c_latent = self.hparams.c_latent
        self.manifold = self.hparams.manifold
        self.dropout = 0.0
        self.weight_decay = 5e-4

        # choose manifold
        self.metric_s = 'euclidean'
        self.rie_pro_input = ToEuclidean()
        if self.hparams.manifold == 'Euclidean':
            self.metric_e = 'euclidean'
            self.rie_pro_latent = ToEuclidean()
        if self.hparams.manifold == 'PoincareBall':
            self.metric_e = 'poin_dist_mobiusm_v2'
            self.rie_pro_latent = ToPoincare(c=self.c_latent, manifold=self.manifold)
        if self.hparams.manifold == 'Hyperboloid':
            self.num_latent_dim += 1
            self.hparams.NetworkStructure[-1] = self.num_latent_dim
            self.metric_e = 'lor_dist_v2'
            self.rie_pro_latent = ToLorentz(c=self.c_latent, manifold=self.manifold)

        self.generator = Generator(dims=self.dims)
        self.classifier = Classifier(num_latent_dim=self.num_latent_dim)
            
        self.Loss = Loss_use_aug.MyLoss(
            v_input=self.hparams.v_input,
            metric_s=self.metric_s,
            metric_e=self.metric_e,
            c_input = self.hparams.c_input,
            c_latent = self.hparams.c_latent,
            pow = self.hparams.pow_latent,
            augNearRate=self.hparams.gamma,
            batchRate=self.hparams.beta,
            )
        
        self.nushadular = nuscheduler.Nushadular(
            nu_start=self.hparams.nu,
            nu_end=self.hparams.ve,
            epoch_start=self.hparams.epochs*2//6,
            epoch_end=self.hparams.epochs*5//6,
            )

        self.criterion = nn.MSELoss()

        print(self.generator)
        print(self.classifier)

    def forward(self, x):
        return self.generator(x)

    def aug_near_mix(self, index, dataset, k=10):
        r = (torch.arange(start=0, end=index.shape[0])*k + torch.randint(low=1, high=k, size=(index.shape[0],)))
        random_select_near_index = dataset.neighbors_index[index.cpu()][:,:k].reshape((-1,))[r]
        random_select_near_data_2 = dataset.data[random_select_near_index]
        random_select_near_data_batch_hot_2 = dataset.batch_hot[random_select_near_index]
        random_rate = torch.rand(size = (index.shape[0], 1)) / 2
        new_data = random_rate*random_select_near_data_2 + (1-random_rate)*dataset.data[index.cpu()]
        new_batch_hot = random_rate*random_select_near_data_batch_hot_2 + (1-random_rate)*dataset.batch_hot[index.cpu()]
        return new_data.to(self.device), new_batch_hot.to(self.device)

    def multi_one_hot(SELF, index_tensor, depth_list):
        one_hot_tensor = F.one_hot(index_tensor[:,0], num_classes=depth_list[0])
        for col in range(1, len(depth_list)):
            next_one_hot = F.one_hot(index_tensor[:,col], num_classes=depth_list[col])
            one_hot_tensor = torch.cat([one_hot_tensor, next_one_hot], 1)

        return one_hot_tensor

    def training_step(self, batch, batch_idx):

        batch_hot, index = batch

        data_1 = self.data_train.data[index].to(self.device)
        data_2, batch_hot_2 = self.aug_near_mix(index, self.data_train, k=self.hparams.K)
        mid_1 = self(data_1)
        mid_2 = self(data_2)
        latent_1 = self.classifier(mid_1)
        latent_2 = self.classifier(mid_2)
        data = torch.cat([data_1, data_2])
        mid = torch.cat((mid_1, mid_2))
        latent = torch.cat([latent_1, latent_2])
        batch_hot = torch.cat((batch_hot, batch_hot_2))

        latent_pro = self.rie_pro_latent(latent)

        loss_gsp, self.dis_p, self.dis_q, self.P, self.Q = self.Loss(
            input_data=mid.reshape(data.shape[0], -1),
            latent_data=latent_pro.reshape(data.shape[0], -1),
            batch_hot = batch_hot,
            rho=0.0,
            sigma=1.0,
            v_latent=self.nushadular.Getnu(self.current_epoch),
        )
        loss = loss_gsp

        return loss

    def validation_step(self, batch, batch_idx):

        batch_hot, index = batch
        data = self.data_train.data[index].to(self.device)
        label = self.data_train.label[index].to(self.device)

        mid = self(data)
        latent = self.classifier(mid)
        latent = self.rie_pro_latent(latent)
        
        return (
            data.detach().cpu().numpy(),
            latent.detach().cpu().numpy(),
            label.detach().cpu().numpy(),
            index.detach().cpu().numpy(),
            )

    def log_dist(self, dist):
        self.dist = dist

    def validation_epoch_end(self, outputs):
        
        if (self.current_epoch+1) % self.log_interval == 0:
            
            data = np.concatenate([ data_item[0] for data_item in outputs ])
            latent = np.concatenate([ data_item[1] for data_item in outputs ])
            label = np.concatenate([ data_item[2] for data_item in outputs ])
            index = np.concatenate([ data_item[3] for data_item in outputs ])
            self.scEmbedding = latent

            if self.hparams.manifold == 'Hyperboloid':
                latent = latent[:, 1:3] / np.expand_dims(1 + latent[:, 0], axis=1)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            {'params': self.generator.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=self.learning_rate, weight_decay=1e-4)
        scheduler = StepLR(opt, step_size=self.hparams.epochs//10, gamma=0.5)

        return [opt], [scheduler]

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        dataset_f = getattr(data_base, self.data_name + "Dataset")
        self.data_train = dataset_f(data_name=self.data_name, batch_class=self.batch_class, raw_data=self.hparams.raw_data, raw_label=self.hparams.raw_label, label_batch=self.hparams.raw_label_batch if self.hparams.raw_label_batch is not None else None)
        self.data_val = self.data_train
        self.data_test = self.data_train
        if stage == 'fit':
            self.data_train._Pretreatment(self.data_train.data, metric_s=self.hparams.metric_s, K=self.hparams.K)

    def train_dataloader(self):
        return  DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return  DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


def dvnet_train(mod: dict) -> dict:
    config = {
        "data_name": "SingleCell",
        "v_input": 100,
        # "nu": 5e-3,
        "ve": -1,
        "NetworkStructure": [-1, 500, 300, 100, 2],
        # "num_latent_dim": 2,
        "pow_latent": 2,
        "K": 5,
        # "gamma": 1000,
        # "beta": 100,
        # "batch_class": 1,
        "metric_s": "euclidean",
        # "manifold": "Hyperboloid",
        "c_input": 1.0,
        "c_latent": 1.0,
        "batch_size": 500,
        "epochs": 200,
        "learning_rate": 1e-3,
        "seed": 1,
        "log_interval": 50,
        # "save_dir": f"save_checkpoint/{mod['data_name']}/",
        # "task_id": "default",
        "raw_data": None,
        "raw_label": None,
        "raw_label_batch": None,
        "gpu": 0,
    }
    config.update(mod)

    pl.utilities.seed.seed_everything(1)

    model = DV_Model(**config)

    trainer = Trainer(
        gpus=config["gpu"],
        max_epochs=config["epochs"],
    )
    trainer.fit(model)
    return model.scEmbedding
