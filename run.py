from pathlib import Path
import yaml
import sys


sys.path.append(str(Path("..").resolve()))
from pipeline import Pipeline

config_path = Path("configs/config.yaml")

print(config_path.resolve())
print(config_path.exists())

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config

pipeline = Pipeline(config)
outputs = pipeline.run()