from typing import Sequence, Any, Sequence, Callable, Dict
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.layers.ilayer import ILayer
from babilim.data import Dataset
from babilim.experiment import Config


class IModel(ILayer):
    def __init__(self, name: str, layer_type: str):
        super().__init__(name, layer_type)

    def fit(self, training_dataset: Dataset, validation_dataset: Dataset, loss, metrics, config: Config):
        if babilim.is_backend(PYTORCH_BACKEND):
            from babilim.models.pt_model import fit as _fit
            _fit(self, training_dataset, validation_dataset, loss, metrics, config)
        elif babilim.is_backend(TF_BACKEND):
            from babilim.models.tf_model import fit as _fit
            _fit(self, training_dataset, validation_dataset, loss, metrics, config)
        else:
            raise NotImplementedError("Unsupported backend: {}".format(babilim.get_backend()))


_MODEL_REGISTRY: Dict[str, Dict[str, IModel]] = defaultdict(dict)

def register_model(name: str, backend: str=None) -> Callable:
    def register_layer_decorator(model):
        if backend is None:
            _MODEL_REGISTRY[PYTORCH_BACKEND][name] = model
            _MODEL_REGISTRY[TF_BACKEND][name] = model
        else:
            _MODEL_REGISTRY[backend][name] = model
        return model
    return register_layer_decorator


def get_model(name: str) -> IModel:
    if name not in _MODEL_REGISTRY[babilim.get_backend()]:
        raise RuntimeError("Layer {} was never registered. Did you forget to import the file in which it gets defined? Or annotating it with @register_layer(...)?".format(name))
    return _MODEL_REGISTRY[babilim.get_backend()][name]
