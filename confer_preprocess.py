"""Stitch the two images in the CONFER dataset together. Then merge the .pts files
"""
import FFMPEGFrames
from pathlib import Path
import numpy as np
from dataset import read_pts, save_pts
from PIL import Image
def main():

    seqpaths = [seq for fold in Path('./data/CONFER/').iterdir() for seq in list(fold.iterdir())]

    for seqpath in seqpaths:
        _seq = [p for p in list(seqpath.iterdir()) if p.name!='stitched']
        assert len(_seq) == 2
        leftseq, rightseq = _seq

        #---------------------------
        # Stitch two images together
        #---------------------------
        leftseq_frames = sorted(list(leftseq.glob('*.jpg')))
        rightseq_frames = sorted(list(rightseq.glob('*.jpg')))

        f = FFMPEGFrames.FFMPEGFrames(str(seqpath))
        f.stitch_images(input_left=leftseq_frames, input_right=rightseq_frames)

        #-----------
        # Fix points
        #-----------
        x_offset, y_offset =  Image.open(leftseq_frames[0]).size

        leftseq_points = sorted(list(leftseq.glob('*.pts')))
        rightseq_points = sorted(list(rightseq.glob('*.pts')))
        for i, (lpath, rpath) in enumerate(zip(leftseq_points, rightseq_points)):
            lpts = read_pts(lpath).squeeze()
            rpts = read_pts(rpath).squeeze()

            rpts[:, 0] += x_offset

            outpath = seqpath.joinpath('stitched', f'{i+1:06}.pts')
            save_pts(outpath, np.vstack([lpts, rpts]))        

if __name__ == '__main__':
    main()