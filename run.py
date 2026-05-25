from pathlib import Path
import sys

import lightning as L
import torch
import yaml

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

sys.path.append(str(Path("..").resolve()))
from pipeline import Pipeline  # noqa: E402


config_path = Path("configs/config.yaml")

print(config_path.resolve())
print(config_path.exists())

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

L.seed_everything(config["training"].get("seed", 42), workers=True)

pipeline = Pipeline(config)
outputs = pipeline.run()