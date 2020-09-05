import pathlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

# Global Variables
SUPPORTED_IMAGE_EXTs = ['npy']
SUPPORTED_POINT_EXTs = ['npy']
ORIG_FPS = 25

class FaceTrackDatasetFolder(Dataset):
    """A dataset for face track images that contained in a folder.
    Each folder must contain all frames of a video in a supported image file and annotations in a supported point file.
    There must be one annotation file for each image file.
    """
    
    def __init__(self, root: str, extensions: [str]) -> None:
        self.root = root

        extensions = [ext.replace(".", "") for ext in extensions]
        paths = [path for ext in extensions
                 for path in pathlib.Path(root).glob(f'**/*.{ext}')]

        unsupported_exts = []
        for i, ext in enumerate(extensions):
            if ext not in SUPPORTED_EXTs:
                unsupported_exts.append(ext)

        if unsupported_exts:
            msg = f"Unsupported extensions are specified: .{' .'.join(unsupported_exts)}"
            raise RuntimeError(msg)
        if len(paths) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(
                    ",".join(extensions))
            raise RuntimeError(msg)

        self.extensions = extensions
        self.paths = paths
        self.x_len = x_len
        self.y_len = y_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x_len = self.x_len
        y_len = self.y_len

        x, y = np.load(self.paths[i], allow_pickle=True)
        # x, y = torch.split(torch.tensor(X, dtype=torch.float32),
        #                    [x_len, y_len], dim=-1)
        return x, y
