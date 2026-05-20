import glob
import os
import random

import lightning as L
import torch
from torch.utils.data import DataLoader

from scs.data_loader import JetDataLoader
from scs.utils import get_label_dict


class DataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()

        self.data_dir = config["path"]
        self.num_datasets = config.get("num_datasets", None)
        self.batch_size = config.get("batch_size", 32)
        self.max_particles_in_jet = config.get("max_particles_in_jet", 128)
        self.chunk_size = config.get("chunk_size", 2048)
        self.samples_per_file = config.get("samples_per_file", 512)
        self.num_workers = config.get("num_workers", 0)

        split_cfg = config.get("split", {})
        self.train_fraction = config.get("train_fraction", split_cfg.get("train", 0.7))
        self.val_fraction = config.get("val_fraction", split_cfg.get("val", 0.15))
        self.test_fraction = config.get("test_fraction", split_cfg.get("test", 0.15))

        self.seed = config.get("seed", 42)

        self.files = None
        self.label_map = None
        self.num_classes = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._setup_done = False

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

    def _split_files(self, files):
        files = list(files)
        n = len(files)

        if n == 0:
            return [], [], []
        if n == 1:
            return files, [], []
        if n == 2:
            return [files[0]], [files[1]], []

        rng = random.Random(self.seed)
        rng.shuffle(files)

        n_train = max(1, int(round(self.train_fraction * n)))
        n_val = max(1, int(round(self.val_fraction * n)))
        n_test = max(1, n - n_train - n_val)

        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        while n_train + n_val + n_test < n:
            n_train += 1

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:n_train + n_val + n_test]

        return train_files, val_files, test_files

    def setup(self, stage=None):
        if self._setup_done:
            return

        if self.files is None:
            self._select_files_once()

        print("\nclasses:")
        self.label_map = get_label_dict(self.files)
        for k, v in self.label_map.items():
            print(v, k)

        self.num_classes = len(self.label_map)

        train_files, val_files, test_files = self._split_files(self.files)

        common_dataset_kwargs = dict(
            max_particles_in_jet=self.max_particles_in_jet,
            chunk_size=self.chunk_size,
            samples_per_file=self.samples_per_file,
            label_map=self.label_map,
            seed=self.seed,
        )

        self.train_dataset = JetDataLoader(
            train_files,
            shuffle_files=True,
            **common_dataset_kwargs,
        )
        self.val_dataset = JetDataLoader(
            val_files,
            shuffle_files=False,
            **common_dataset_kwargs,
        )
        self.test_dataset = JetDataLoader(
            test_files,
            shuffle_files=False,
            **common_dataset_kwargs,
        )

        print(f"\nDataset sizes:")
        print(f"Train: {len(self.train_dataset)}")
        print(f"Val:   {len(self.val_dataset)}")
        print(f"Test:  {len(self.test_dataset)}")

        self._setup_done = True

    def _loader_kwargs(self, shuffle: bool):
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,  # IterableDataset handles ordering itself
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=shuffle,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return kwargs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))