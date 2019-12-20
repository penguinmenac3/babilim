import sys
import os
import numpy as np
import babilim
import babilim.logger as logger
from babilim.experiment import import_checkpoint_config
from babilim import PYTORCH_BACKEND, TF_BACKEND, PHASE_VALIDATION, info, error, warn, status

from examples.fashion_mnist import FashionMnistDataset
from examples.fashion_mnist import FashionMnistConfig


def evaluate_checkpoint(config: FashionMnistConfig, checkpoint_folder: str, validation_dataset: FashionMnistDataset):
    info("Evaluating checkpoint: {}".format(checkpoint_folder))
    # Load model
    model_path = os.path.join(checkpoint_folder, "src", "examples")
    info("Importing model from: {}".format(model_path))
    sys.path.append(model_path)
    from fashion_mnist import FashionMnistModel
    model = FashionMnistModel(config)
    sys.path.remove(model_path)

    # Init model on gpu
    feat, _ = validation_dataset[0]
    model.predict(**feat._asdict())

    # Load weights
    tmp = os.path.join(checkpoint_folder, "checkpoints")
    checkpoint = os.path.join(tmp, sorted(os.listdir(tmp))[-1])
    info("Loading weights from: {}".format(checkpoint))
    model.load(checkpoint)

    # Evaluate
    correct = 0
    wrong = 0
    info("Evaluating {} examples.".format(len(validation_dataset)))
    for feat, label in validation_dataset:
        preds = model.predict(**feat._asdict())
        pred_digit = np.argmax(preds.class_id)
        true_digit = label.class_id
        if pred_digit == true_digit:
            correct += 1
        else:
            wrong += 1
        status("Accuracy: {:.1f}%  ({} vs {}), {} == {}".format(correct/(correct+wrong) * 100, correct, wrong, true_digit, pred_digit), end="")
    status("Accuracy: {:.1f}%  ({} vs {})".format(correct/(correct+wrong) * 100, correct, wrong))


if __name__ == "__main__":
    babilim.set_backend(PYTORCH_BACKEND)

    checkpoint = os.path.join("checkpoints", sorted(os.listdir("checkpoints"))[-1])
    config: FashionMnistConfig = import_checkpoint_config(os.path.join(checkpoint, "src", "examples", "fashion_mnist.py"))

    validation_dataset = FashionMnistDataset(config, PHASE_VALIDATION)
    evaluate_checkpoint(config, checkpoint, validation_dataset)