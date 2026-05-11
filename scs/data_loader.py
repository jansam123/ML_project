import uproot
import awkward as ak
import numpy as np
import torch
import os

from scs.utils import get_label_dict


class JetDataLoader:
    def __init__(self, files: list, max_particles_in_jet: int = 128):
        self.files = files
        self.label_map = get_label_dict(files)
        self.max_particles_in_jet = max_particles_in_jet
        self.trees = {
            file: uproot.open(file)["tree"]
            for file in self.files
        }

        self.index = []
        for file in self.files:
            n = self.trees[file].num_entries
            self.index.extend([(file, idx) for idx in range(n)])

    def __len__(self):
        return len(self.index)

    def _label_from_file(self, file: str):
        name = os.path.basename(file)

        key = "_".join(name.split("_")[:-1])

        return self.label_map[key]

    def _build_constituent_features(self, event):

        px = ak.to_numpy(event.part_px)
        py = ak.to_numpy(event.part_py)

        pt = np.sqrt(px**2 + py**2)

        x = np.stack([
            pt,
            ak.to_numpy(event.part_deta),
            ak.to_numpy(event.part_dphi),
            ak.to_numpy(event.part_charge),
            ak.to_numpy(event.part_isChargedHadron),
            ak.to_numpy(event.part_isNeutralHadron),
            ak.to_numpy(event.part_isPhoton),
            ak.to_numpy(event.part_isElectron),
            ak.to_numpy(event.part_isMuon),
        ], axis=1)

        return x.astype(np.float32)


    def _build_jet_features(self, event):

        jet_features = np.array([
            event.jet_pt,
            event.jet_eta,
            event.jet_phi,
            event.jet_energy,
            event.jet_sdmass,
            event.jet_tau1,
            event.jet_tau2,
            event.jet_tau3,
            event.jet_tau4,
        ], dtype=np.float32)

        return jet_features

    def __getitem__(self, idx):

        file, event_idx = self.index[idx]


        tree = self.trees[file]
        event = tree.arrays(
            library="ak",
            entry_start=event_idx,
            entry_stop=event_idx + 1,
        )[0]


        x = self._build_constituent_features(event)
        jet_features = self._build_jet_features(event)

        number_of_particles = x.shape[0]
        number_of_features = x.shape[1]

        number_of_particles_used = min(
            number_of_particles,
            self.max_particles_in_jet
        )

        x_padding = np.zeros(
            (self.max_particles_in_jet, number_of_features),
            dtype=np.float32
        )

        mask = np.zeros(
            self.max_particles_in_jet,
            dtype=np.float32
        )

        x_padding[:number_of_particles_used] = \
            x[:number_of_particles_used]

        mask[:number_of_particles_used] = 1.0

        y = self._label_from_file(file)

        return {
            "x": torch.tensor(x_padding, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "jet_features": torch.tensor(
                jet_features,
                dtype=torch.float32
            ),
            "y": torch.tensor(y, dtype=torch.long),
        }