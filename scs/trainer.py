import torch
import torch.nn as nn
import lightning as L


class Trainer(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.model(
            batch["x"],
            batch["mask"],
            batch["jet_features"],
        )

    def _shared_step(self, batch):
        y = batch["y"]
        logits = self(batch)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.config.get("lr", 1e-3)
        return torch.optim.Adam(self.parameters(), lr=lr)

