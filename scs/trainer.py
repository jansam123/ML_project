import lightning as L
import torch
import torch.nn as nn


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
        return loss, acc, y

    def training_step(self, batch, batch_idx):
        loss, acc, y = self._shared_step(batch)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=y.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, y = self._shared_step(batch)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=y.size(0),
        )

        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_epoch=True,
            batch_size=y.size(0),
        )

        return loss

    def on_test_start(self):
        self.test_probs = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        loss, acc, y = self._shared_step(batch)

        logits = self(batch)
        probs = torch.softmax(logits, dim=1)

        self.test_probs.append(probs.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            batch_size=y.size(0),
        )

        self.log(
            "test_acc",
            acc,
            on_epoch=True,
            batch_size=y.size(0), 
        )

        return loss

    def configure_optimizers(self):
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        epochs = self.config.get("epochs", 100)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }