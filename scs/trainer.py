import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0

    def fit(self, model, train_loader, val_loader=None):
        model.to(self.device)

        epochs = self.config["epochs"]

        for epoch in range(epochs):
            self.epoch = epoch

            print(f"\nEpoch [{epoch + 1}/{epochs}]")

            self._train_epoch(model, train_loader)

            if val_loader is not None:
                self._validate_epoch(model, val_loader)

    def _train_epoch(self, model, loader):

        # NOTE: Forward-pass-only skeleton.
        # No loss / backward / optimizer step yet.

        model.train()

        progress_bar = tqdm(
            loader,
            desc="Training",
            leave=False
        )

        for batch in progress_bar:

            x = self._extract_input(batch).to(self.device)

            with torch.no_grad():
                _ = model(x)

    def _validate_epoch(self, model, loader):
        model.eval()

        progress_bar = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        with torch.no_grad():
            for batch in progress_bar:
                x = self._extract_input(batch).to(self.device)
                _ = model(x)

    def test(self, model, test_loader):
        model.eval()

        outputs = []

        progress_bar = tqdm(
            test_loader,
            desc="Testing"
        )

        with torch.no_grad():
            for batch in progress_bar:
                x = self._extract_input(batch).to(self.device)
                out = model(x)
                outputs.append(out)

        return outputs

    def _extract_input(self, batch):

        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, (tuple, list)):
            return batch[0]

        if isinstance(batch, dict):
            return batch.get("x", None)

        raise ValueError("Unsupported batch format")