"""Alpha VAE performs a weighted sum between two loss functions, given by
`loss = alpha * loss_func1 + (1 - alpha) * loss_func2`"""


import os
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchinfo import summary

from lvae3d.LightningVAETrainers import LightningVAE_alpha
from lvae3d.models.ResNetVAE_3D import ResNet18_3DVAE
from lvae3d.util.DataLoaders import Dataset3D, DataModule
from lvae3d.util.LossFunctions import SpectralLoss3D
from lvae3d.util.MetadataDicts import MetadataAlpha


if __name__ == '__main__':

    # Hyperparameters
    PATCH_SIZE = 64
    N_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    LATENT_DIM = 512
    ALPHA = 0.999
    N_CHANNELS = 3
    PARALLEL = False
    AMSGRAD = True
    vae = ResNet18_3DVAE(LATENT_DIM, N_CHANNELS)
    loss_func1 = nn.BCEWithLogitsLoss()
    loss_func2 = SpectralLoss3D()

    # Filepaths
    TRAIN_DIR = './data3d_norm/d3d_3d_train/'
    VAL_DIR = './data3d_norm/d3d_3d_test/'
    OUT_DIR = f'./out/{vae.__class__.__name__}/alphaVAE_latent{LATENT_DIM}/'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Load checkpoint parameters
    load_model = False
    checkpoint_path = './out/ResNet18_3DVAE/alphaVAE_latent512/checkpoint_weights_epoch10.ckpt'

    # Metadata
    metadata = MetadataAlpha()
    if load_model:
        metadata.load(f'{OUT_DIR}metadata.yaml')
        metadata.metadata_dict['initial_epoch'] = metadata.metadata_dict['n_epochs'] + 1
        metadata.metadata_dict['n_epochs'] += N_EPOCHS
    else:
        metadata.create(vae=vae,
                        loss_func1=loss_func1,
                        loss_func2=loss_func2,
                        parallel=PARALLEL,
                        patch_size=PATCH_SIZE,
                        n_channels = N_CHANNELS,
                        n_epochs=N_EPOCHS,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        batch_size=BATCH_SIZE,
                        latent_dim=LATENT_DIM,
                        alpha=ALPHA,
                        amsgrad=AMSGRAD
                        )

    # Load microstructure data
    seed_everything(42, workers=True)
    train_dataset = Dataset3D(root_dir=TRAIN_DIR)
    val_dataset = Dataset3D(root_dir=VAL_DIR)
    data_module = DataModule(BATCH_SIZE, train_dataset, val_dataset, num_workers=8)

    # Train model
    torch.set_float32_matmul_precision('medium')
    if load_model:
        model = LightningVAE_alpha.load_from_checkpoint(checkpoint_path,
                                                        vae=vae,
                                                        metadata=metadata,
                                                        loss_func1=loss_func1,
                                                        loss_func2=loss_func2
                                                        )
    else:
        model = LightningVAE_alpha(vae, metadata, loss_func1, loss_func2)
    summary(model, (1, 3, 64, 64, 64))

    checkpoint_callback = ModelCheckpoint(dirpath=OUT_DIR)
    trainer = Trainer(min_epochs=N_EPOCHS,
                      max_epochs=N_EPOCHS,
                      default_root_dir=OUT_DIR,
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(model, data_module)
    trainer.save_checkpoint(filepath=f'{OUT_DIR}checkpoint_weights_epoch{N_EPOCHS}.ckpt', weights_only=True)

    metadata.save(f'{OUT_DIR}metadata.yaml')
