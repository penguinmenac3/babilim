from typing import Sequence, Any, Sequence, Callable, Dict
import babilim
from babilim.layers.ilayer import ILayer


class IModel(ILayer):
    def __init__(self, name: str, layer_type: str):
        super().__init__(name, layer_type)
