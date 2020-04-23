import os
import tarfile as tar
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple


class MultiTarReader(object):
    def __init__(self, *, tarfiles: List[str] = None, tarfile: str = None) -> None:
        """
        Create a tar dataset_class from either a single string or a list of strings pointing to tar files.

        :param tarfiles: A list of files to use for this dataset_class.
        """
        # Normalize parameters
        if tarfiles is None and tarfile is not None:
            tarfiles = [tarfile]
        elif tarfile is None and tarfiles is None:
            raise RuntimeError("You must either specify zipfile or zipfiles.")
        tarfiles = [os.path.normpath(x) for x in tarfiles]

        # Open zip files
        self.__tarfiles = [tar.open(filepath) for filepath in tarfiles]

        # Populate filelist and fileinfos
        self.__infos = []
        self.__filelist = []
        for i, z in enumerate(self.__tarfiles):
            tar_name = tarfiles[i].split(os.sep)[-1]
            data = {}
            for x in z.getmembers():
                data[tar_name + os.sep + os.path.normpath(x.name)] = x
                self.__filelist.append(tar_name + os.sep + os.path.normpath(x.name))
            self.__infos.append(data)

        # Populate filetree
        self.__filetree = {}
        for filename in self.__filelist:
            tmp = filename.split(os.sep)
            folders = tmp[:-1]
            filename = tmp[-1]
            tmp = self.__filetree
            for f in folders:
                if f not in tmp:
                    tmp[f] = {}
                tmp = tmp[f]
            tmp[filename] = {}

    @property
    def file_list(self) -> List[str]:
        """
        The list of all files in all tar files.
        :return: A list containing all files.
        """
        return self.__filelist

    @property
    def file_tree(self) -> Dict:
        """
        The tree structure of all files in all tar files.
        :return:
        """
        return self.__filetree

    def __find_file(self, filename: str) -> Tuple[int, object]:
        """
        Find a file in all tars.
        :param filename: The sample_token to search.
        :return: In which tar it is findable with what fileinfo.
        """
        tarid = -1
        for i, inf in enumerate(self.__infos):
            if filename in inf.keys():
                tarid = i
                break
        if tarid < 0:
            raise RuntimeError("Cannot find file: {}".format(filename))
        return tarid, self.__infos[tarid][filename]

    def read(self, filename: str) -> str:
        """
        Read a files content as a string.
        :param filename: The sample_token of the file.
        :return: The content of the file as a string.
        """
        zipid, fileinfo = self.__find_file(filename)
        with self.__tarfiles[zipid].extractfile(fileinfo) as f:
            return f.read()

    def read_image(self, filename: str) -> np.ndarray:
        """
        Read an image from the tars.
        :param filename: The sample_token of the file.
        :return: The image as a numpy array.
        """
        zipid, fileinfo = self.__find_file(filename)
        with self.__tarfiles[zipid].extractfile(fileinfo) as f:
            return np.array(Image.open(f))

    def open_file(self, filename: str) -> object:
        """
        Open a file with a file handle so user can decide what to do with it.
        :param filename: The file which to open.
        :return: The file handle of the file (like open(...)).
        """
        zipid, fileinfo = self.__find_file(filename)
        return self.__tarfiles[zipid].extractfile(fileinfo)
