from babilim.data import Dataset
from babilim.experiment import Config


def fit(model, training_dataset: Dataset, validation_dataset: Dataset, loss, metrics, config: Config):
    batched_dataset = training_dataset.to_pytorch()