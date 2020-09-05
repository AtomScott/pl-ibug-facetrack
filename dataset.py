import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from typing import Union
from PIL import Image
from torchvision.transforms import ToTensor

# Global Variables
SUPPORTED_IMAGE_EXTs = ['npy']
SUPPORTED_POINT_EXTs = ['npy']
ORIG_FPS = 25

class FaceTrackDatasetFolder(Dataset):
    """A dataset for face track images that contained in a folder.
    Each folder must contain all frames of a video in a supported image file and annotations in a supported point file.
    There must be one annotation file for each image file.
    """
    
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path

        try:
            p = Path(root_path)
            frame_paths = sorted(list(p.glob('**/*.jpg')) + list(p.glob('**/*.png')))
            annot_paths = sorted(list(p.glob('**/*.pts')))

            assert len(frame_paths) == len(annot_paths), f"{root_path}: {len(frame_paths)} != {len(annot_paths)}"

        except AssertionError() as ae:
            error_count += 1
            print(f"{root_path} has unmatching number of frames and annotations.")

            frame_paths = []
            annot_paths = []


        self.frame_paths = frame_paths
        self.annot_paths = annot_paths
        self.transforms = ToTensor()
                
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, i):
        x = pil_loader(self.frame_paths[i])
        x = self.transforms(x)

        _pts = read_pts(self.annot_paths[i])
        y = []
        for pts in _pts:
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            y.append([xmin,ymin, xmax,ymax])
        y = torch.tensor(y)

        return x, y

    @staticmethod
    def plot(x, y):

        fig,ax = plt.subplots(1)
        ax.imshow(x)
        for xmin,ymin, xmax,ymax,  in y:
            rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.show()
        
def read_pts(filename):
    x = np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))
    x = x.reshape(-1, 68, 2)
    return x


def save_pts(filename, x):
    return np.savetxt(filename, x, comments=("version:", "n_points:", "{", "}"))

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
