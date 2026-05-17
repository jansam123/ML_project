import os
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import Dataset

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

class JetDataLoader(Dataset):
    def __init__(self, files: list, max_particles_in_jet: int = 128):
        super().__init__()

        self.files = list(files)
        self.max_particles_in_jet = max_particles_in_jet
        self.label_map = get_label_dict(self.files)
        self.num_classes = len(self.label_map)
        self.branches = branches

        # per-worker cache (safe with multiprocessing)
        self._trees: Dict[str, object] = {}


        # build index WITHOUT keeping ROOT handles open
        # This creates a dict that keeps track of how many entries each file has
        # This lets __getitem__ quickly determine which file and event to load without rescanning ROOT files during training.
        self.index: List[Tuple[str, int]] = []
        for f in self.files:
            with uproot.open(f) as fh:
                n = fh["tree"].num_entries
            self.index.extend((f, i) for i in range(n))

    def __len__(self):
        return len(self.index)

    def _label_from_file(self, file: str) -> int:
        name = os.path.basename(file)
        key = "_".join(name.split("_")[:-1])
        return int(self.label_map[key])

    def _get_tree(self, file: str):
        # This method returns a tree object from a ROOT file
        # It only opens the file the first time
        if file not in self._trees:
            self._trees[file] = uproot.open(file)["tree"]
        return self._trees[file]

    def _load_event(self, file: str, event_idx: int):
        # This method loads exactly one event (jet) from one ROOT file
        tree = self._get_tree(file)
        arrays = tree.arrays(
            self.branches,
            entry_start=event_idx,
            entry_stop=event_idx + 1,
            library="ak",
        )
        return arrays[0]

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

    def __getitem__(self, idx):
        file, event_idx = self.index[idx]

        event = self._load_event(file, event_idx)
        x = self._build_constituent_features(event)
        jet_features = self._build_jet_features(event)
        n = min(x.shape[0], self.max_particles_in_jet)
        x_pad = np.zeros((self.max_particles_in_jet, x.shape[1]), dtype=np.float32)
        mask = np.zeros(self.max_particles_in_jet, dtype=np.float32)
        x_pad[:n] = x[:n]
        mask[:n] = 1.0
        y = self._label_from_file(file)

        return {
            "x": torch.tensor(x_pad),
            "mask": torch.tensor(mask),
            "jet_features": torch.tensor(jet_features),
            "y": torch.tensor(y, dtype=torch.long),
        }
