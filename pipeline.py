from scs.data_module import DataModule
from scs.model import Model
from scs.trainer import Trainer

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.data_module = DataModule(config["data"])
        self.model = Model(config["model"])
        self.trainer = Trainer(config["training"])

    def run(self):
        self.data_module.setup()

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        self.trainer.fit(self.model, train_loader, val_loader)

        test_loader = self.data_module.test_dataloader()
        return self.trainer.test(self.model, test_loader)