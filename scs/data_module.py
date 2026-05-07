import os
import glob
import random

import torch
from torch.utils.data import random_split, DataLoader

import lightning as L

from scs.data_loader import JetDataLoader
from scs.utils import get_label_dict

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_datasets: int | None = None,
        batch_size: int = 32,
        max_particles_in_jet: int = 128,
        num_workers: int = 0,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        seed: int =42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_datasets = num_datasets
        self.batch_size = batch_size
        self.max_particles_in_jet = max_particles_in_jet
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.seed = seed

    def setup(self, stage=None):

        files = glob.glob(os.path.join(self.data_dir, "*.root"))
        if self.num_datasets:
            files = random.sample(files, k=self.num_datasets)

        print("\classes:")
        for k, v in get_label_dict(files).items():
            print(v, k)

        full_dataset = JetDataLoader(
            files=files,
            max_particles_in_jet=self.max_particles_in_jet,
        )

        n_total = len(full_dataset)
        n_train = int(self.train_fraction * n_total)
        n_val = int(self.val_fraction * n_total)
        n_test = n_total - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

        print(f"\nDataset sizes:")
        print(f"Train: {len(self.train_dataset)}")
        print(f"Val:   {len(self.val_dataset)}")
        print(f"Test:  {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )