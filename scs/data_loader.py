import os
from typing import Dict

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
    File-level streaming dataset with chunk-level preprocessing.

    Main speedup:
    - read a whole chunk from ROOT
    - convert/pad/stack the whole chunk at once
    - yield per-event tensors from prebuilt NumPy arrays

    This removes most of the per-event awkward -> NumPy conversion overhead.
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
        self.max_particles_in_jet = int(max_particles_in_jet)
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

    def _pad_jagged(self, arr, fill_value=0.0):
        """
        Pad/clip a jagged awkward array to [n_events, max_particles_in_jet].
        """
        arr = ak.pad_none(arr, self.max_particles_in_jet, clip=True)
        arr = ak.fill_none(arr, fill_value)
        return ak.to_numpy(arr, allow_missing=False).astype(np.float32, copy=False)

    def _build_constituent_features_chunk(self, arrays):
        """
        Build:
          x:    (B, Nmax, 9)
          mask: (B, Nmax)
        for an entire chunk at once.
        """
        px = arrays["part_px"]
        py = arrays["part_py"]
        pt = np.sqrt(px * px + py * py)

        feature_arrays = [
            pt,
            arrays["part_deta"],
            arrays["part_dphi"],
            arrays["part_charge"],
            arrays["part_isChargedHadron"],
            arrays["part_isNeutralHadron"],
            arrays["part_isPhoton"],
            arrays["part_isElectron"],
            arrays["part_isMuon"],
        ]

        padded_features = [self._pad_jagged(a, fill_value=0.0) for a in feature_arrays]
        x = np.stack(padded_features, axis=-1).astype(np.float32, copy=False)

        counts = np.asarray(ak.to_numpy(ak.num(px, axis=1)), dtype=np.int64)
        counts = np.minimum(counts, self.max_particles_in_jet)
        mask = (
            np.arange(self.max_particles_in_jet)[None, :] < counts[:, None]
        ).astype(np.float32)

        return x, mask

    def _build_jet_features_chunk(self, arrays):
        """
        Build:
          jet_features: (B, 9)
        for an entire chunk at once.
        """
        jet_features = np.stack(
            [
                self._to_numpy_1d(arrays["jet_pt"]),
                self._to_numpy_1d(arrays["jet_eta"]),
                self._to_numpy_1d(arrays["jet_phi"]),
                self._to_numpy_1d(arrays["jet_energy"]),
                self._to_numpy_1d(arrays["jet_sdmass"]),
                self._to_numpy_1d(arrays["jet_tau1"]),
                self._to_numpy_1d(arrays["jet_tau2"]),
                self._to_numpy_1d(arrays["jet_tau3"]),
                self._to_numpy_1d(arrays["jet_tau4"]),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        return jet_features

    def _yield_chunk_samples(self, file: str, arrays):
        """
        Convert one chunk into NumPy arrays once, then yield per-event samples.
        """
        x, mask = self._build_constituent_features_chunk(arrays)
        jet_features = self._build_jet_features_chunk(arrays)

        y_value = self._label_from_file(file)
        y_tensor = torch.tensor(y_value, dtype=torch.long)

        n_chunk = x.shape[0]
        for i in range(n_chunk):
            yield {
                "x": torch.from_numpy(x[i]),
                "mask": torch.from_numpy(mask[i]),
                "jet_features": torch.from_numpy(jet_features[i]),
                "y": y_tensor,
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

                    yield from self._yield_chunk_samples(file, arrays)