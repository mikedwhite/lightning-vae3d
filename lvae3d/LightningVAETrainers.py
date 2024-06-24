import torch
from torch import nn
import lightning as L
from statistics import fmean

from lvae3d.util.LossFunctions import KLDivergence
from lvae3d.util.LossFunctions import SpectralLoss3D


class LightningVAE_alpha(L.LightningModule):
    def __init__(self, vae, metadata, loss_func1=nn.BCEWithLogitsLoss(), loss_func2=SpectralLoss3D):
        """Lightning trainer for VAE. Loss is computed as a weighted sum between two loss functions, such that
        `loss = alpha * loss_func1 + (1 - alpha) * loss_func2`.

        Parameters
        ----------
        vae : lightning.LightningModule
        metadata : lvae3d.util.Metadata
            Metadata instance from the util.MetadataDicts module.
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        """
        super().__init__()
        self.metadata = metadata
        if self.metadata.metadata_dict['parallel'] is True:
            self.vae = nn.DataParallel(vae)
        else:
            self.vae = vae
        self.loss_func1 = loss_func1
        self.loss_func2 = loss_func2
        self.lr = self.metadata.metadata_dict['learning_rate']
        self.weight_decay = self.metadata.metadata_dict['weight_decay']
        self.alpha = self.metadata.metadata_dict['alpha']
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        x_hat, mu, log_sigma, z = self.vae(x)
        return x_hat, mu, log_sigma, z

    def training_step(self, train_batch):
        x = train_batch['image']
        x_hat, _, _, _ = self.vae(x)
        loss1 = self.loss_func1(x, x_hat)
        loss2 = self.loss_func2(x, x_hat)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        self.train_loss += [loss.item()]
        if self.metadata.metadata_dict['parallel'] is True:
            self.log_dict({'train_loss1': loss1, 'train_loss2': loss2, 'train_loss': loss},
                          on_epoch=True, on_step=False, sync_dist=True)
        else:
            self.log_dict({'train_loss1': loss1, 'train_loss2': loss2, 'train_loss': loss},
                          on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self.metadata.metadata_dict['train_loss'] += [fmean(self.train_loss)]
        self.train_loss = []

    def validation_step(self, val_batch):
        x = val_batch['image']
        x_hat, _, _, _ = self.vae(x)
        loss1 = self.loss_func1(x, x_hat)
        loss2 = self.loss_func2(x, x_hat)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        self.val_loss += [loss.item()]
        if self.metadata.metadata_dict['parallel'] is True:
            self.log_dict({'val_loss1': loss1, 'val_loss2': loss2, 'val_loss': loss},
                          on_epoch=True, on_step=False, sync_dist=True)
        else:
            self.log_dict({'val_loss1': loss1, 'val_loss2': loss2, 'val_loss': loss},
                          on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self.metadata.metadata_dict['val_loss'] += [fmean(self.val_loss)]
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.metadata.metadata_dict['amsgrad']
                                     )
        return [optimizer]


class LightningVAE_beta(L.LightningModule):
    def __init__(self, vae, metadata, loss_func=nn.MSELoss()):
        """Lightning trainer for VAE. Loss is computed as a weighted sum between two loss functions, such that
        `loss = loss_func + beta * KLDivergence`.

        Parameters
        ----------
        vae : lightning.LightningModule
        metadata : lvae3d.util.Metadata
            Metadata instance from the util.MetadataDicts module.
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        """
        super().__init__()
        self.metadata = metadata
        if self.metadata.metadata_dict['parallel'] is True:
            self.vae = nn.DataParallel(vae)
        else:
            self.vae = vae
        self.loss_func = loss_func
        self.lr = self.metadata.metadata_dict['learning_rate']
        self.weight_decay = self.metadata.metadata_dict['weight_decay']
        self.beta = self.metadata.metadata_dict['beta']
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        x_hat, mu, log_sigma, z = self.vae(x)
        return x_hat, mu, log_sigma, z

    def training_step(self, train_batch):
        x = train_batch['image']
        x_hat, mu, log_sigma, _ = self.vae(x)
        loss1 = self.loss_func(x, x_hat)
        loss2 = KLDivergence()(mu, log_sigma)
        loss = loss1 + self.beta * loss2
        self.train_loss += [loss.item()]
        if self.metadata.metadata_dict['parallel'] is True:
            self.log_dict({'train_loss1': loss1, 'train_loss_kl': loss2, 'train_loss': loss},
                          on_epoch=True, on_step=False, sync_dist=True)
        else:
            self.log_dict({'train_loss1': loss1, 'train_loss_kl': loss2, 'train_loss': loss},
                          on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self.metadata.metadata_dict['train_loss'] += [fmean(self.train_loss)]
        self.train_loss = []

    def validation_step(self, val_batch):
        x = val_batch['image']
        x_hat, mu, log_sigma, _ = self.vae(x)
        loss1 = self.loss_func(x, x_hat)
        loss2 = KLDivergence()(mu, log_sigma)
        loss = loss1 + self.beta * loss2
        self.val_loss += [loss.item()]
        if self.metadata.metadata_dict['parallel'] is True:
            self.log_dict({'val_loss1': loss1, 'val_loss_kl': loss2, 'val_loss': loss},
                          on_epoch=True, on_step=False, sync_dist=True)
        else:
            self.log_dict({'val_loss1': loss1, 'val_loss_kl': loss2, 'val_loss': loss},
                          on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self.metadata.metadata_dict['val_loss'] += [fmean(self.val_loss)]
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.metadata.metadata_dict['amsgrad'],
                                     )
        return [optimizer]
