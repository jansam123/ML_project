import torch


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0

    def fit(self, model, train_loader, val_loader=None):
        model.to(self.device)

        for epoch in range(self.config["epochs"]):
            self.epoch = epoch

            self._train_epoch(model, train_loader)

            if val_loader is not None:
                self._validate_epoch(model, val_loader)

    def _train_epoch(self, model, loader):

        # NOTE: This is currently a forward-pass-only skeleton.
        # No loss computation, backpropagation, or optimizer step is performed here.
        # Intended for pipeline validation (data/model wiring), not training.
        # To enable learning, this loop must be extended with:
        # - loss function
        # - backward pass (loss.backward())
        # - optimizer step (optimizer.step())

        model.train()
        for batch in loader:
            
            x = self._extract_input(batch)

            with torch.no_grad():
                _ = model(x)

    def _validate_epoch(self, model, loader):
        model.eval()

        with torch.no_grad():
            for batch in loader:
                x = self._extract_input(batch)
                _ = model(x)

    def test(self, model, test_loader):
        model.eval()

        outputs = []

        with torch.no_grad():
            for batch in test_loader:
                x = self._extract_input(batch)
                out = model(x)
                outputs.append(out)

        return outputs

    def _extract_input(self, batch):
        """
        Assumes batch is either:
        - tensor
        - (x, y)
        - dict with 'x'
        """
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, (tuple, list)):
            return batch[0]

        if isinstance(batch, dict):
            return batch.get("x", None)

        raise ValueError("Unsupported batch format")