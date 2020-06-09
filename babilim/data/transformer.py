class Transformer(object):
    """
    A transformer should implement ``__call__``.
    """
    def __call__(self, *args):
        """
        This function gets the data from the previous transformer or dataset as input and should output the data again.
        :param args: The input data.
        :return: The output data.
        """
        raise NotImplementedError

    @property
    def version(self):
        """
        Defines the version of the transformer. The name can be also something descriptive of the method.

        :return: The version number of the transformer.
        """
        raise NotImplementedError
