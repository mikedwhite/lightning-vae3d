import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class Dataset3D(Dataset):
    """Dataset of 3D volume elements. Requires volume elements to be stored as PyTorch tensor (.pt) files.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = sorted(glob.glob(f'{self.root_dir}*.pt'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]
        rve = torch.load(filename, weights_only=True)
        sample = {'filename': os.path.basename(filename),
                  'image': rve}

        return sample


class DataModule(L.LightningDataModule):
    def __init__(self, batchsize, train_dataset, val_dataset=None, test_dataset=None, num_workers=4):
        super(DataModule, self).__init__()
        self.batch_size = batchsize
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True, persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=True, persistent_workers=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers,
                                pin_memory=True, persistent_workers=True)
        return test_loader
