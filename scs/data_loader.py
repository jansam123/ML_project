import os
from typing import Dict, List

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import IterableDataset, get_worker_info

from scs.utils import get_label_dict


branches = [
            "part_px",
            "part_py",
            "part_deta",
            "part_dphi",
            "part_charge",
            "part_isChargedHadron",
            "part_isNeutralHadron",
            "part_isPhoton",
            "part_isElectron",
            "part_isMuon",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_energy",
            "jet_sdmass",
            "jet_tau1",
            "jet_tau2",
            "jet_tau3",
            "jet_tau4",
        ]


class JetDataLoader(IterableDataset):
    """
    File-level streaming dataset.

    - No global event index.
    - Files are split across workers.
    - Each worker opens a file with a context manager, reads chunks, then closes it.
    - Each file contributes at most `samples_per_file` samples per epoch.
    """

    def __init__(
        self,
        files: list,
        max_particles_in_jet: int = 128,
        chunk_size: int = 2048,
        samples_per_file: int = 512,
        label_map: dict | None = None,
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.files = list(files)
        self.max_particles_in_jet = max_particles_in_jet
        self.chunk_size = max(1, int(chunk_size))
        self.samples_per_file = max(1, int(samples_per_file))
        self.shuffle_files = shuffle_files
        self.seed = seed

        self.label_map = label_map if label_map is not None else get_label_dict(self.files)
        self.num_classes = len(self.label_map)

        self.branches = branches
        # Metadata only: count entries per file once.
        self.file_event_counts: Dict[str, int] = {}
        for f in self.files:
            with uproot.open(f) as fh:
                self.file_event_counts[f] = fh["tree"].num_entries

    def __len__(self):
        return sum(
            min(self.file_event_counts[f], self.samples_per_file)
            for f in self.files
        )

    def _label_from_file(self, file: str) -> int:
        name = os.path.basename(file)
        key = "_".join(name.split("_")[:-1])

        if key not in self.label_map:
            raise KeyError(
                f"Label key '{key}' not found. Available keys: {list(self.label_map.keys())}"
            )

        y = int(self.label_map[key])
        if not (0 <= y < self.num_classes):
            raise ValueError(
                f"Invalid label {y}. Must be in [0, {self.num_classes - 1}]"
            )
        return y

    @staticmethod
    def _to_numpy_1d(x):
        return np.asarray(ak.to_numpy(x), dtype=np.float32)

    @staticmethod
    def _to_float(x):
        v = np.asarray(ak.to_numpy(x))
        return float(v.reshape(-1)[0])

    def _build_constituent_features(self, event):
        px = self._to_numpy_1d(event["part_px"])
        py = self._to_numpy_1d(event["part_py"])
        pt = np.sqrt(px**2 + py**2)

        return np.stack(
            [
                pt,
                self._to_numpy_1d(event["part_deta"]),
                self._to_numpy_1d(event["part_dphi"]),
                self._to_numpy_1d(event["part_charge"]),
                self._to_numpy_1d(event["part_isChargedHadron"]),
                self._to_numpy_1d(event["part_isNeutralHadron"]),
                self._to_numpy_1d(event["part_isPhoton"]),
                self._to_numpy_1d(event["part_isElectron"]),
                self._to_numpy_1d(event["part_isMuon"]),
            ],
            axis=1,
        ).astype(np.float32)

    def _build_jet_features(self, event):
        return np.array(
            [
                self._to_float(event["jet_pt"]),
                self._to_float(event["jet_eta"]),
                self._to_float(event["jet_phi"]),
                self._to_float(event["jet_energy"]),
                self._to_float(event["jet_sdmass"]),
                self._to_float(event["jet_tau1"]),
                self._to_float(event["jet_tau2"]),
                self._to_float(event["jet_tau3"]),
                self._to_float(event["jet_tau4"]),
            ],
            dtype=np.float32,
        )

    def _make_sample(self, file: str, event) -> dict:
        x = self._build_constituent_features(event)
        jet_features = self._build_jet_features(event)

        n = min(x.shape[0], self.max_particles_in_jet)

        x_pad = np.zeros((self.max_particles_in_jet, x.shape[1]), dtype=np.float32)
        mask = np.zeros(self.max_particles_in_jet, dtype=np.float32)

        if n > 0:
            x_pad[:n] = x[:n]
            mask[:n] = 1.0

        y = self._label_from_file(file)

        return {
            "x": torch.from_numpy(x_pad),
            "mask": torch.from_numpy(mask),
            "jet_features": torch.from_numpy(jet_features),
            "y": torch.tensor(y, dtype=torch.long),
        }

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
            files = self.files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = self.files[worker_id::num_workers]

        rng = np.random.default_rng(self.seed + worker_id)

        if self.shuffle_files:
            files = list(files)
            rng.shuffle(files)

        for file in files:
            n_events = self.file_event_counts[file]
            n_take = min(n_events, self.samples_per_file)

            if n_take <= 0:
                continue

            if n_events > n_take:
                start0 = int(rng.integers(0, n_events - n_take + 1))
            else:
                start0 = 0

            with uproot.open(file) as fh:
                tree = fh["tree"]

                stop0 = start0 + n_take
                for start in range(start0, stop0, self.chunk_size):
                    stop = min(start + self.chunk_size, stop0)

                    arrays = tree.arrays(
                        self.branches,
                        entry_start=start,
                        entry_stop=stop,
                        library="ak",
                    )

                    n_chunk = len(arrays[self.branches[0]])
                    for i in range(n_chunk):
                        event = {k: arrays[k][i] for k in self.branches}
                        yield self._make_sample(file, event)