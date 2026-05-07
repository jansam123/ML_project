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

        self.index = []
        for file in self.files:
            n = uproot.open(file)["tree"].num_entries
            self.index.extend([(file, idx) for idx in range(n)])

    def __len__(self):
        # len of JetDataLoader is now len of index
        return len(self.index)

    def _label_from_file(self, file: str):
        name = os.path.basename(file)
        # remove _000.root etc.
        key = "_".join(name.split("_")[:-1])
        return self.label_map[key]

    def _build_features(self, event):
        # we can always add more features to this method 
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

    def __getitem__(self, idx):
        # __getitem__ makes sure we can acces the data like this: 
        # class_instance[index] instead of class_instance.method(index)
        file, event_idx = self.index[idx]
        tree = uproot.open(file)["tree"]

        # we use library="ak" (awkward) because we have variable data sizes
        event = tree.arrays(library="ak", entry_start=event_idx, entry_stop=event_idx + 1)[0]
        x = self._build_features(event) 

        number_of_particles = x.shape[0]
        number_of_features = x.shape[1]
        number_of_particles_used = min(number_of_particles, self.max_particles_in_jet)

        # first we create zero-arrays: padding and a mask
        x_padding = np.zeros((self.max_particles_in_jet, number_of_features), dtype=np.float32)
        mask = np.zeros(self.max_particles_in_jet, dtype=np.float32)

        # the padding is either the real lenght or a specified max lenght         
        x_padding[:number_of_particles_used] = x[:number_of_particles_used]
        # the mask tells us if there is a artificial padding or a real particle (we need this for fixed-size tensors)
        mask[:number_of_particles_used] = 1.0 

        # Now we can get the labels from the file names (maybe use the data instead, if possible?)
        y = self._label_from_file(file)

        return {
            "x": torch.tensor(x_padding, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }
