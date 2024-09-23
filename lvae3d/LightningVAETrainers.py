import torch
from torch import nn
import lightning as L
from statistics import fmean

from lvae3d.util.LossFunctions import KLDivergence, SpectralLoss3D


class VAETrainerAlpha(L.LightningModule):
    def __init__(self, vae, metadata, loss_func1=nn.BCEWithLogitsLoss(), loss_func2=SpectralLoss3D):
        """Lightning trainer for ResNet18_3DVAE. Loss is computed as a weighted sum between two loss functions, such
        that `loss = alpha * loss_func1 + (1 - alpha) * loss_func2`.

        Parameters
        ----------
        metadata : lvae3d.util.MetadataDicts.MetadataAlpha
            Metadata instance from the util.MetadataDicts module.
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        """
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func1', 'loss_func2'])
        self.metadata = metadata
        # TODO: add error checking for vae and loss functions input compared to metadata entries
        if self.metadata.metadata_dict['parallel'] is True:
            self.vae = nn.DataParallel(vae(self.metadata.metadata_dict['latent_dim'],
                                           self.metadata.metadata_dict['n_channels'],
                                           self.metadata.metadata_dict['hidden_dim']
                                           ))
        else:
            self.vae = vae(self.metadata.metadata_dict['latent_dim'],
                           self.metadata.metadata_dict['n_channels'],
                           self.metadata.metadata_dict['hidden_dim']
                           )
        self.loss_func1 = loss_func1
        self.loss_func2 = loss_func2
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
        loss = self.metadata.metadata_dict['alpha'] * loss1 + (1 - self.metadata.metadata_dict['alpha']) * loss2
        self.train_loss += [loss.item()]
        self.log_dict({'train_loss1': loss1, 'train_loss2': loss2, 'train_loss': loss}, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self.metadata.metadata_dict['train_loss'] += [fmean(self.train_loss)]
        self.train_loss = []

    def validation_step(self, val_batch):
        x = val_batch['image']
        x_hat, _, _, _ = self.vae(x)
        loss1 = self.loss_func1(x, x_hat)
        loss2 = self.loss_func2(x, x_hat)
        loss = self.metadata.metadata_dict['alpha'] * loss1 + (1 - self.metadata.metadata_dict['alpha']) * loss2
        self.val_loss += [loss.item()]
        self.log_dict({'val_loss1': loss1, 'val_loss2': loss2, 'val_loss': loss}, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self.metadata.metadata_dict['val_loss'] += [fmean(self.val_loss)]
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.metadata.metadata_dict['learning_rate'],
                                     weight_decay=self.metadata.metadata_dict['weight_decay'],
                                     amsgrad=self.metadata.metadata_dict['amsgrad']
                                     )
        return [optimizer]


class VAETrainerBeta(L.LightningModule):
    def __init__(self, vae, metadata, loss_func=nn.MSELoss()):
        """Lightning trainer for ResNet18_3DVAE. Loss is computed as a weighted sum between two loss functions, such
        that `loss = loss_func + beta * KLDivergence`.

        Parameters
        ----------
        metadata : lvae3d.util.MetadataDicts.MetadataBeta
            Metadata instance from the util.MetadataDicts module.
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        """
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func1', 'loss_func2'])
        self.metadata = metadata
        if self.metadata.metadata_dict['parallel'] is True:
            self.vae = nn.DataParallel(vae(self.metadata.metadata_dict['latent_dim'],
                                           self.metadata.metadata_dict['n_channels'],
                                           self.metadata.metadata_dict['hidden_dim']
                                           ))
        else:
            self.vae = vae(self.metadata.metadata_dict['latent_dim'],
                           self.metadata.metadata_dict['n_channels'],
                           self.metadata.metadata_dict['hidden_dim']
                           )
        self.loss_func = loss_func
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
        loss = loss1 + self.metadata.metadata_dict['beta'] * loss2
        self.train_loss += [loss.item()]
        self.log_dict({'train_loss1': loss1, 'train_loss_kl': loss2, 'train_loss': loss}, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self.metadata.metadata_dict['train_loss'] += [fmean(self.train_loss)]
        self.train_loss = []

    def validation_step(self, val_batch):
        x = val_batch['image']
        x_hat, mu, log_sigma, _ = self.vae(x)
        loss1 = self.loss_func(x, x_hat)
        loss2 = KLDivergence()(mu, log_sigma)
        loss = loss1 + self.metadata.metadata_dict['beta'] * loss2
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
                                     lr=self.metadata.metadata_dict['learning_rate'],
                                     weight_decay=self.metadata.metadata_dict['weight_decay'],
                                     amsgrad=self.metadata.metadata_dict['amsgrad'],
                                     )
        return [optimizer]


class VAETrainerAlphaBeta(L.LightningModule):
    def __init__(self, vae, metadata, loss_func1=nn.BCEWithLogitsLoss(), loss_func2=SpectralLoss3D):
        """Lightning trainer for ResNet34_3DVAE. Loss is computed as a weighted sum between two loss functions, such
        that `loss = alpha * loss_func1 + (1 - alpha) * loss_func2 + beta * KLDivergence`.

        Parameters
        ----------
        metadata : lvae3d.util.MetadataDicts.MetadataAlphaBeta
            Metadata instance from the util.MetadataDicts module.
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        """
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func1', 'loss_func2'])
        self.metadata = metadata
        if self.metadata.metadata_dict['parallel'] is True:
            self.vae = nn.DataParallel(vae(self.metadata.metadata_dict['latent_dim'],
                                           self.metadata.metadata_dict['n_channels'],
                                           self.metadata.metadata_dict['hidden_dim']
                                           ))
        else:
            self.vae = vae(self.metadata.metadata_dict['latent_dim'],
                           self.metadata.metadata_dict['n_channels'],
                           self.metadata.metadata_dict['hidden_dim']
                           )
        self.loss_func1 = loss_func1
        self.loss_func2 = loss_func2
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        x_hat, mu, log_sigma, z = self.vae(x)
        return x_hat, mu, log_sigma, z

    def training_step(self, train_batch):
        x = train_batch['image']
        x_hat, mu, log_sigma, _ = self.vae(x)
        loss1 = self.loss_func1(x, x_hat)
        loss2 = self.loss_func2(x, x_hat)
        loss3 = KLDivergence()(mu, log_sigma)
        loss = self.metadata.metadata_dict['alpha'] * loss1 + (1 - self.metadata.metadata_dict['alpha']) * loss2 + self.metadata.metadata_dict['beta'] * loss3
        self.train_loss += [loss.item()]
        self.log_dict({'train_loss1': loss1, 'train_loss2': loss2, 'kl_div':loss3, 'train_loss': loss}, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self.metadata.metadata_dict['train_loss'] += [fmean(self.train_loss)]
        self.train_loss = []

    def validation_step(self, val_batch):
        x = val_batch['image']
        x_hat, mu, log_sigma, _ = self.vae(x)
        loss1 = self.loss_func1(x, x_hat)
        loss2 = self.loss_func2(x, x_hat)
        loss3 = KLDivergence()(mu, log_sigma)
        loss = self.metadata.metadata_dict['alpha'] * loss1 + (1 - self.metadata.metadata_dict['alpha']) * loss2 + self.metadata.metadata_dict['beta'] * loss3
        self.val_loss += [loss.item()]
        self.log_dict({'val_loss1': loss1, 'val_loss2': loss2, 'kl_div':loss3, 'val_loss': loss}, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self.metadata.metadata_dict['val_loss'] += [fmean(self.val_loss)]
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.metadata.metadata_dict['learning_rate'],
                                     weight_decay=self.metadata.metadata_dict['weight_decay'],
                                     amsgrad=self.metadata.metadata_dict['amsgrad']
                                     )
        return [optimizer]
