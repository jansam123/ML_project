from pathlib import Path
import sys

import yaml
import torch
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(".").resolve()))

from scs.data_module import DataModule
from scs.model import *
from scs.trainer import Trainer



CONFIG_PATH = "configs/config.yaml"  # !!!!!!!!!!!!!! CONFIG PATH: change model (jetmodel.yaml, mixedmodel.yaml, ecc.)

CHECKPOINT_PATH = "lightning_logs/version_28/checkpoints/epoch=69-step=40880.ckpt"  # !!!!!!!!! CHECKPOINT PATH: cchange .ckpt of model

OUTPUT_PATH = "plots/confusion_matrix.png"  # !!!!! OUTPUT PATH

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

L.seed_everything(config["training"].get("seed", 42), workers=True)





data_module = DataModule(config["data"])
data_module.setup()

model_config = dict(config["model"])
model_config["num_classes"] = data_module.num_classes

# BUILD MODEL


match model_config["name"]:
    case "JetOnlyModel":
        model = JetOnlyModel(model_config)

    case "ParticleOnlyModel":
        model = ParticleOnlyModel(model_config)

    case "HybridModel":
        model = HybridModel(model_config)

    case "ParticleTransformer":
        model = ParticleTransformer(model_config)

    case "particle_mlp":
        model = Model(model_config)

    case _:
        raise ValueError(f"Unknown model name: {model_config['name']}")


lightning_model = Trainer.load_from_checkpoint(
    CHECKPOINT_PATH,
    model=model,
    config=config["training"],
)

pl_trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    precision="16-mixed",
    logger=False,
    enable_progress_bar=True,
)

pl_trainer.test(
    lightning_model,
    datamodule=data_module,
)


if not hasattr(lightning_model, "test_probs") or len(lightning_model.test_probs) == 0:
    raise RuntimeError("No test predictions found. Did test_step run correctly?")

probs = torch.cat(lightning_model.test_probs, dim=0)
targets = torch.cat(lightning_model.test_targets, dim=0)

preds = torch.argmax(probs, dim=1)

# CLASS NAMES

label_map = data_module.label_map

class_names = [None] * len(label_map)
for class_name, class_index in label_map.items():
    class_names[class_index] = class_name

# CONFUSION MATRIX

cm = confusion_matrix(
    targets.numpy(),
    preds.numpy(),
    labels=list(range(len(class_names))),
)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)


output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title(f"Confusion Matrix - {model_config['name']}")

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nConfusion matrix saved here:\n{output_path}")