import uproot
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class RootJetDataset(Dataset):
    def __init__(self, filename, tree_name="tree"):
        self.feature_branches = [
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_energy",
            "jet_nparticles",
            "jet_sdmass",
            "jet_tau1",
            "jet_tau2",
            "jet_tau3",
            "jet_tau4",
        ]

        self.label_branches = [
            "label_QCD",
            "label_Hbb",
            "label_Hcc",
            "label_Hgg",
            "label_H4q",
            "label_Hqql",
            "label_Zqq",
            "label_Wqq",
            "label_Tbqq",
            "label_Tbl",
        ]

        with uproot.open(filename) as f:
            tree = f[tree_name]

            # Convert to Numpy
            arrays = tree.arrays(
                self.feature_branches + self.label_branches,
                library="np",
            )

        self.X = np.column_stack([arrays[br] for br in self.feature_branches])

        self.Y = np.column_stack([arrays[br] for br in self.label_branches])


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]          
        y = self.Y[idx]          

        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(y, dtype=torch.float32)

        return x, y


if __name__ == "__main__":
    filename = "/home/scur0034/ML_project/data/JetClass/Pythia/train_100M/HToBB_000.root"
    tree_name = "tree"  

    dataset = RootJetDataset(filename, tree_name)
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
    )

    for batch_x, batch_y in dataloader:
        print(batch_x.shape, batch_y.shape)
        # batch_x: [B, 10]   (10 jet-features)
        # batch_y: [B, 10]   (10 label-branches)
        break

    print(batch_x)

values = np.array([row[0] for row in batch_x])
print(len(values))

#plt.hist(values, bins=50, edgecolor='black', alpha=0.7)
#plt.show()

# check if it is consistent with explore_data.py
print(f"  Mean: {np.mean(values):.4f}")
print(f"  Std: {np.std(values):.4f}")
print(f"  Min: {np.min(values):.4f}")
print(f"  Max: {np.max(values):.4f}")
print(f"  Total entries: {len(values)}")