import glob
import os
import random

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from scs.data_loader import JetDataLoader
from scs.utils import get_label_dict


class DataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()

        self.data_dir = config["path"]
        self.num_datasets = config.get("num_datasets", None)
        self.batch_size = config.get("batch_size", 32)
        self.max_particles_in_jet = config.get("max_particles_in_jet", 128)
        self.num_workers = config.get("num_workers", 0)
        self.train_fraction = config.get("train_fraction", 0.8)
        self.val_fraction = config.get("val_fraction", 0.1)
        self.seed = config.get("seed", 42)

        self.files = None
        self.num_classes = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _select_files_once(self):
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.root")))

        if self.num_datasets is not None:
            if self.num_datasets > len(all_files):
                raise ValueError(
                    f"num_datasets={self.num_datasets} exceeds available ROOT files={len(all_files)}"
                )
            rng = random.Random(self.seed)
            self.files = rng.sample(all_files, k=self.num_datasets)
        else:
            self.files = all_files

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        if self.files is None:
            self._select_files_once()

        print("\nclasses:")
        label_map = get_label_dict(self.files)
        for k, v in label_map.items():
            print(v, k)

        full_dataset = JetDataLoader(
            files=self.files,
            max_particles_in_jet=self.max_particles_in_jet,
        )

        self.num_classes = full_dataset.num_classes

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
            pin_memory_device="cuda",
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4,
            drop_last=True,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4,
            drop_last=False,
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4,
            drop_last=False,
        )