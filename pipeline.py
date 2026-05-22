import lightning as L

from scs.data_module import DataModule
from scs.model import *
from scs.trainer import Trainer


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.data_module = DataModule(config["data"])

        self.model = None
        self.lightning_model = None

        self.pl_trainer = L.Trainer(
            # max_steps=50_000,
            max_epochs=config["training"]["epochs"],
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

    def run(self):
        self.data_module.setup()

        model_config = dict(self.config["model"])
        model_config["num_classes"] = self.data_module.num_classes

        match model_config["name"]:
            case "JetOnlyModel":
                self.model = JetOnlyModel(model_config)
            case "ParticleOnlyModel":
                self.model = ParticleOnlyModel(model_config)
            case "HybridModel":
                self.model = HybridModel(model_config)
            case "ParticleTransformer":
                self.model = ParticleTransformer(model_config)
            case "particle_mlp":
                self.model = Model(model_config)
            case _:
                raise ValueError(f"Unknown model: {model_config['name']}")

        self.lightning_model = Trainer(self.model, self.config["training"])

        self.pl_trainer.fit(
            self.lightning_model,
            datamodule=self.data_module,
        )

        test_metrics = self.pl_trainer.test(
            self.lightning_model,
            datamodule=self.data_module,
        )

        probs_list = self.lightning_model.test_probs
        targets_list = self.lightning_model.test_targets

        if len(probs_list) == 0:
            raise RuntimeError("No test predictions collected. Did test_step run?")

        probs = torch.cat(probs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        return {
            "metrics": test_metrics,
            "probs": probs,        # shape (N, C)
            "targets": targets,    # shape (N,)
        }