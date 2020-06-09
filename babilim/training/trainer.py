from babilim.data import Dataloader


class Trainer(object):
    def run_epoch(self, dataloader: Dataloader, phase: str, epoch: int):
        raise NotImplementedError()

    def fit(self, train_dataloader: Dataloader, dev_dataloader: Dataloader, epochs: int):
        raise NotImplementedError()
