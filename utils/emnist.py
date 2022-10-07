

from .mnist import MNIST
import string
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import shutil
from .utils import *

class EMNIST(MNIST):
    """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
            and  ``EMNIST/processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
    # Merged Classes assumes Same structure for both uppercase and lowercase version
    _merged_classes = {'c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z'}
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        'byclass': sorted(list(_all_classes)),
        'bymerge': sorted(list(_all_classes - _merged_classes)),
        'balanced': sorted(list(_all_classes - _merged_classes)),
        'letters': ['N/A'] + list(string.ascii_lowercase),
        'digits': list(string.digits),
        'mnist': list(string.digits),
    }

    def __init__(self, root: str, split: str, **kwargs: Any) -> None:
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)
        self.classes = self.classes_split_dict[self.split]

    @staticmethod
    def _training_file(split) -> str:
        return 'training_{}.pt'.format(split)

    @staticmethod
    def _test_file(split) -> str:
        return 'test_{}.pt'.format(split)

    @property
    def _file_prefix(self) -> str:
        return f"emnist-{self.split}-{'train' if self.train else 'test'}"

    @property
    def images_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-images-idx3-ubyte")

    @property
    def labels_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-labels-idx1-ubyte")

    def _load_data(self):
        return read_image_file(self.images_file), read_label_file(self.labels_file)

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file, self.labels_file))

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        download_and_extract_archive(self.url, download_root=self.raw_folder, md5=self.md5)
        gzip_folder = os.path.join(self.raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                extract_archive(os.path.join(gzip_folder, gzip_file), self.raw_folder)
        shutil.rmtree(gzip_folder)