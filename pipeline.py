from scs.data_module import DataModule
from scs.model import Model
from scs.trainer import Trainer


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.data_module = DataModule(config["data"])
        self.model = Model(config["model"])
        self.trainer = Trainer(config["training"])

    def run(self):
        print("start set up")
        self.data_module.setup()
        print("complete set up")

        print("start train loader")
        train_loader = self.data_module.train_dataloader()
        print("end train loader")

        print("start val loader")
        val_loader = self.data_module.val_dataloader()
        print("end val loader")

        print("start train loader")
        self.trainer.fit(self.model, train_loader, val_loader)
        print("end train loader")

        print("start test loader")
        test_loader = self.data_module.test_dataloader()
        print("end test loader")

        return self.trainer.test(self.model, test_loader)