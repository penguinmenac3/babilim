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
import os
from typing import Dict, Any
import json
from importlib import import_module
import inspect

_sentinel = object()


class ConfigPart(object):
    """
    Converts a dictionary into an object.
    """

    def __init__(self, **kwargs):
        """
        Create an object from a dictionary.

        :param d: The dictionary to convert.
        """
        self.immutable = False
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict:
        dictionary = dict((key, value.to_dict()) if isinstance(value, ConfigPart) else (key, value)
                          for (key, value) in self.__dict__.items())
        del dictionary["immutable"]
        return dictionary

    def __repr__(self) -> str:
        return "ConfigPart(" + self.__str__() + ")"

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    def get(self, key: str, default: Any = _sentinel) -> Any:
        """
        Get the value specified in the dictionary or a default.
        :param key: The key which should be retrieved.
        :param default: The default that is returned if the key is not set.
        :return: The value from the dict or the default.
        """
        if default is _sentinel:
            default = ConfigPart()
        return self.__dict__[key] if key in self.__dict__ else default

    def __getitem__(self, key: str) -> Any:
        """
        Get the value specified in the dictionary or a dummy.
        :param key: The key which should be retrieved.
        :return: The value from the dict or a dummy.
        """
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if "immutable" not in self.__dict__:  # In case users might not call constructor
            self.__dict__["immutable"] = False
        if self.immutable:
            raise RuntimeError("Trying to modify hyperparameters outside constructor.")

        if isinstance(value, str):
            # Try to match linux path style with anything that matches
            for env_var in list(os.environ.keys()):
                s = "$" + env_var
                value = value.replace(s, os.environ[env_var].replace("\\", "/"))

            # Try to match windows path style with anything that matches
            for env_var in list(os.environ.keys()):
                s = "%" + env_var + "%"
                value = value.replace(s, os.environ[env_var].replace("\\", "/"))

            if "%" in value or "$" in value:
                raise RuntimeError("Cannot resove all environment variables used in: '{}'".format(value))
        super.__setattr__(self, key, value)

    def __eq__(self, other: 'ConfigPart') -> bool:
        if not isinstance(other, ConfigPart):
            # don't attempt to compare against unrelated types
            return NotImplemented

        for k in self.__dict__:
            if not k in other.__dict__:
                return False
            if not self.__dict__[k] == other.__dict__[k]:
                return False

        for k in other.__dict__:
            if not k in self.__dict__:
                return False

        return True


class Config(ConfigPart):
    def __init__(self) -> None:
        """
        A configuration for a deep learning project.

        This class should never be instantiated directly, subclass it instead.

        The following parameters are set by default and should be changed after calling super.
        
            train_batch_size = 1
            train_experiment_name = None
            train_checkpoint_path = "checkpoints"
            train_epochs = 50
            train_log_steps = 100
            train_learning_rate_shedule = None
            train_optimizer = None
            arch_prepare = None
            arch_model = None
            arch_loss = None
            arch_metrics = None
            problem_base_dir = None
            problem_dataset = None

        You can add further attributes by simply adding them.
        """
        # Training parameters.
        self.train_batch_size = 1
        self.train_experiment_name = None
        self.train_checkpoint_path = "checkpoints"
        self.train_epochs = 50
        self.train_log_steps = 100
        self.train_learning_rate_shedule = None
        self.train_optimizer = None

        # Architectural parameters (like preparing the data, the model, the loss and then some metrics)
        self.arch_prepare = None
        self.arch_model = None
        self.arch_loss = None
        self.arch_metrics = None

        # Required for general dataset loading. (Non architecture specific.)
        self.problem_base_dir = None
        self.problem_dataset = None

        super().__init__()

    def __repr__(self) -> str:
        return "Config(" + self.__str__() + ")"

    @staticmethod
    def __has_attribute(obj: object, name: str) -> bool:
        """
        Checks if the object has an attribute.

        :param obj: The object that should be checked.
        :param name: The attribute that should be found.
        :return: True if the object has the attribute, False otherwise.
        """
        return name in obj.__dict__ and obj.__dict__[name] is not None

    def check_completness(self) -> bool:
        """
        Check the config for completeness.

        This method checks for the common bare minimum.
        If it fails to find something it throws an assertion error.
        :return: True if no exception occurs.
        """
        # Check for training parameters
        assert Config.__has_attribute(self, "train_experiment_name")
        assert Config.__has_attribute(self, "train_checkpoint_path")
        assert Config.__has_attribute(self, "train_batch_size")
        assert Config.__has_attribute(self, "train_epochs")
        assert Config.__has_attribute(self, "train_log_steps")
        assert Config.__has_attribute(self, "train_optimizer")
        assert Config.__has_attribute(self, "train_learning_rate_shedule")
        assert Config.__has_attribute(self, "arch_prepare")
        assert Config.__has_attribute(self, "arch_model")
        assert Config.__has_attribute(self, "arch_loss")
        assert Config.__has_attribute(self, "arch_metrics")
        assert Config.__has_attribute(self, "problem_base_dir")
        assert Config.__has_attribute(self, "problem_dataset")

        return True


def import_config(config_file: str) -> Config:
    """
    Only libraries should use this method. Human users should directly import their configs.
    Automatically imports the most specific config from a given file.

    :param config_file: The configuration file which should be loaded.
    :return: The configuration object.
    """
    module_name = config_file.replace("\\", ".").replace("/", ".").replace(".py", "")
    module = import_module(module_name)
    symbols = list(module.__dict__.keys())
    symbols = [x for x in symbols if not x.startswith("__")]
    n = None
    for x in symbols:
        if not inspect.isclass(module.__dict__[x]):  # in Case we found something that is not a class ignore it.
            continue
        if issubclass(module.__dict__[x], Config):
            # Allow multiple derivatives of config, when they are derivable from each other in any direction.
            if n is not None and not issubclass(module.__dict__[x], module.__dict__[n]) and not issubclass(
                    module.__dict__[n], module.__dict__[x]):
                raise RuntimeError(
                    "You must only have one class derived from Config in {}. It cannot be decided which to use.".format(
                        config_file))
            # Pick the most specific one if they can be derived.
            if n is None or issubclass(module.__dict__[x], module.__dict__[n]):
                n = x
    if n is None:
        raise RuntimeError("There must be at least one class in {} derived from Config.".format(config_file))
    config = module.__dict__[n]()
    return config
