import sys
import traceback
from typing import Iterable, Iterator, Any

from babilim.core.tensor import TensorWrapper


class Dataloader(Iterable):
    def __init__(self, native_dataloader, dataset):
        self.dataset = dataset
        self._tensor_wrapper = TensorWrapper()
        self.native_dataloader = native_dataloader

    def __iter__(self) -> Iterator:
        class TensorDataloaderIterator(Iterator):
            def __init__(self, native_dataloader, tensor_wrapper):
                self._tensor_wrapper = tensor_wrapper
                self.native_dataloader_iter = iter(native_dataloader)

            def __next__(self) -> Any:
                # Print index errors, they probably were an error and not intentional.
                try:
                    x, y = next(self.native_dataloader_iter)
                    inp = dict(x._asdict())
                    outp = dict(y._asdict())
                    inp = self._tensor_wrapper.wrap(inp)
                    outp = self._tensor_wrapper.wrap(outp)
                    inp = type(x)(**inp)
                    outp = type(y)(**outp)
                    return inp, outp
                except IndexError as e:
                    traceback.print_exc(file=sys.stderr)
                    raise e
        return TensorDataloaderIterator(self.native_dataloader, self._tensor_wrapper)

    def __len__(self) -> int:
        return len(self.native_dataloader)
