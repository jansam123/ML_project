import uproot
import numpy as np
import awkward as ak


class RootFileReader:
    CONSTITUENT_BRANCHES = [
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
    ]

    JET_BRANCHES = [
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

    def __init__(self, path: str):
        self.path = path
        self.tree = uproot.open(path)["tree"]

        self.constituents = self.tree.arrays(
            self.CONSTITUENT_BRANCHES,
            library="ak"
        )

        self.jet_features = self.tree.arrays(
            self.JET_BRANCHES,
            library="np"
        )

        self.n_events = len(self.jet_features["jet_pt"])

    def get_event(self, idx: int):
        px = ak.to_numpy(self.constituents["part_px"][idx])
        py = ak.to_numpy(self.constituents["part_py"][idx])

        pt = np.sqrt(px ** 2 + py ** 2)

        x = np.stack([
            pt,
            ak.to_numpy(self.constituents["part_deta"][idx]),
            ak.to_numpy(self.constituents["part_dphi"][idx]),
            ak.to_numpy(self.constituents["part_charge"][idx]),
            ak.to_numpy(self.constituents["part_isChargedHadron"][idx]),
            ak.to_numpy(self.constituents["part_isNeutralHadron"][idx]),
            ak.to_numpy(self.constituents["part_isPhoton"][idx]),
            ak.to_numpy(self.constituents["part_isElectron"][idx]),
            ak.to_numpy(self.constituents["part_isMuon"][idx]),
        ], axis=1).astype(np.float32)

        jet = np.stack([
            self.jet_features["jet_pt"][idx],
            self.jet_features["jet_eta"][idx],
            self.jet_features["jet_phi"][idx],
            self.jet_features["jet_energy"][idx],
            self.jet_features["jet_sdmass"][idx],
            self.jet_features["jet_tau1"][idx],
            self.jet_features["jet_tau2"][idx],
            self.jet_features["jet_tau3"][idx],
            self.jet_features["jet_tau4"][idx],
        ]).astype(np.float32)

        return x, jet