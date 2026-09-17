import hydra
import lightning as L
from typing import Optional
from omegaconf import ValueNode, DictConfig
from torch_geometric.data import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(self, datasets: DictConfig, num_workers: ValueNode, batch_size: ValueNode):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_dataset = hydra.utils.instantiate(self.datasets.valid)

        if stage is None or stage == "test":
            self.test_dataset = hydra.utils.instantiate(self.datasets.test)

    def train_dataloader(self, shuffle = True):
        return DataLoader(self.train_dataset, shuffle=shuffle, batch_size=self.batch_size.train, num_workers=self.num_workers)

    def val_dataloader(self, shuffle = False):
        return DataLoader(self.val_dataset, shuffle=shuffle, batch_size=self.batch_size.valid, num_workers=self.num_workers)

    def test_dataloader(self, shuffle = False):
        return DataLoader(self.test_dataset, shuffle=shuffle, batch_size=self.batch_size.test, num_workers=self.num_workers)