# MIT License
#
# Copyright (c) 2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from abc import ABC, abstractmethod


class ClassDecorator(ABC):
    def __get__(self, obj, objtype):
        """Support instance methods."""
        import functools
        return functools.partial(self.__call__, obj)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class RunOnlyOnce(ClassDecorator):
    def __init__(self, f):
        """
        A decorator that ensures a function in an object gets only called exactly once.

        :param f: The function that should be wrapped.
        """
        self.f = f
        self.called = {}

    def __call__(self, *args, **kwargs):
        if args[0] not in self.called:
            self.called[args[0]] = True
            self.f(*args, **kwargs)