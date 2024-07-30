"""Alpha VAE performs a weighted sum between two loss functions, given by
`loss = alpha * loss_func1 + (1 - alpha) * loss_func2`"""


import os

import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torchinfo import summary

from lvae3d.LightningVAETrainers import ResNet18VAE_alpha
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
    TrainerModule = ResNet18VAE_alpha
    loss_func1 = nn.BCEWithLogitsLoss()
    loss_func2 = SpectralLoss3D()

    # Filepaths
    TRAIN_DIR = '/path/to/train/data/'
    VAL_DIR = '/path/to/val/data/'
    OUT_DIR = f'./out/{TrainerModule.__name__}/latent{LATENT_DIM}/'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Load checkpoint parameters
    load_model = False
    checkpoint_path = '/path/to/checkpoint.ckpt'

    # Torch config
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Metadata
    metadata = MetadataAlpha()
    if load_model:
        metadata.load(f'{OUT_DIR}metadata.yaml')
        metadata.metadata_dict['initial_epoch'] = metadata.metadata_dict['n_epochs'] + 1
        metadata.metadata_dict['n_epochs'] += N_EPOCHS
    else:
        metadata.create(TrainerModule=TrainerModule,
                        loss_func1=loss_func1,
                        loss_func2=loss_func2,
                        alpha=ALPHA,
                        parallel=PARALLEL,
                        patch_size=PATCH_SIZE,
                        n_channels = N_CHANNELS,
                        n_epochs=N_EPOCHS,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        batch_size=BATCH_SIZE,
                        latent_dim=LATENT_DIM,
                        amsgrad=AMSGRAD
                        )

    # Load microstructure data
    train_dataset = Dataset3D(root_dir=TRAIN_DIR)
    val_dataset = Dataset3D(root_dir=VAL_DIR)
    data_module = DataModule(BATCH_SIZE, train_dataset, val_dataset, num_workers=8)

    # Train model
    if load_model:
        model = TrainerModule.load_from_checkpoint(checkpoint_path,
                                                   metadata=metadata,
                                                   loss_func1=loss_func1,
                                                   loss_func2=loss_func2
                                                   )
    else:
        model = TrainerModule(metadata, loss_func1, loss_func2)
    summary(model, (1, 3, 64, 64, 64))

    checkpoint_callback = ModelCheckpoint(dirpath=OUT_DIR)
    trainer = Trainer(min_epochs=N_EPOCHS,
                      max_epochs=N_EPOCHS,
                      default_root_dir=OUT_DIR,
                      callbacks=[checkpoint_callback],
                      fast_dev_run=True,
                      accelerator='cpu'
                      )
    trainer.fit(model, data_module)
    trainer.save_checkpoint(filepath=f'{OUT_DIR}checkpoint_weights_epoch{N_EPOCHS}.ckpt', weights_only=True)

    metadata.save(f'{OUT_DIR}metadata.yaml')
