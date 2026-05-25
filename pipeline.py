from pathlib import Path
from datetime import datetime
import json
import yaml
import torch
import lightning as L

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from scs.data_module import DataModule
from scs.model import (
    JetOnlyModel,
    ParticleOnlyModel,
    HybridModel,
    ParticleTransformer,
)
from scs.trainer import Trainer


class Pipeline:
    def __init__(self, config):
        self.config = config

        # create run directory
        config_name = config["model"]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = (
            Path("runs")
            / config_name
            / timestamp
        )

        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRun directory:")
        print(self.run_dir.resolve())

        # save config
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        # logger
        self.logger = CSVLogger(
            save_dir=str(self.run_dir),
            name="logs",
        )

        # checkpoints
        monitor = (
            config["training"]
            .get("monitor", "val_loss")
        )

        monitor_mode = (
            config["training"]
            .get("monitor_mode", "min")
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.run_dir / "checkpoints",
            filename="best",
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=1,
            save_last=True,
        )

        # data
        self.data_module = DataModule(config["data"])

        self.model = None
        self.lightning_model = None

        # trainer
        self.pl_trainer = L.Trainer(
            logger=self.logger,
            callbacks=[self.checkpoint_callback],

            max_epochs=config["training"]["epochs"],

            accelerator="auto",
            devices="auto",

            precision="16-mixed",

            log_every_n_steps=10,
            enable_progress_bar=True,
        )

    def _build_model(self):
        model_config = dict(self.config["model"])
        model_config["num_classes"] = self.data_module.num_classes

        match model_config["name"]:

            case "JetOnlyModel":
                return JetOnlyModel(model_config)

            case "ParticleOnlyModel":
                return ParticleOnlyModel(model_config)

            case "HybridModel":
                return HybridModel(model_config)

            case "ParticleTransformer":
                return ParticleTransformer(model_config)

            case _:
                raise ValueError(
                    f"Unknown model: {model_config['name']}"
                )

    def run(self):

        # setup datasets
        self.data_module.setup()

        # build model
        self.model = self._build_model()

        self.lightning_model = Trainer(
            self.model,
            self.config["training"],
        )

        # train
        self.pl_trainer.fit(
            self.lightning_model,
            datamodule=self.data_module,
        )

        # test
        test_metrics = self.pl_trainer.test(
            self.lightning_model,
            datamodule=self.data_module,
        )

        probs = torch.cat(
            self.lightning_model.test_probs,
            dim=0,
        )

        targets = torch.cat(
            self.lightning_model.test_targets,
            dim=0,
        )

        # save outputs
        torch.save(
            probs,
            self.run_dir / "probs.pt",
        )

        torch.save(
            targets,
            self.run_dir / "targets.pt",
        )

        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(
                test_metrics,
                f,
                indent=2,
            )

        # save final model weights
        torch.save(
            self.model.state_dict(),
            self.run_dir / "model.pt",
        )

        print("\nSaved:")
        print(self.run_dir / "model.pt")
        print(self.run_dir / "probs.pt")
        print(self.run_dir / "targets.pt")
        print(self.run_dir / "metrics.json")
        print(self.run_dir / "logs")

        return {
            "metrics": test_metrics,
            "probs": probs,
            "targets": targets,
            "run_dir": str(self.run_dir),
        }