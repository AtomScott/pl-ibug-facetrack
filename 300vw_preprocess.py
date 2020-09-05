"""Script to split .avi videos into frames.
"""
import FFMPEGFrames
from pathlib import Path

def main():
    videopaths = Path('./data/300VW').glob('**/*.avi')

    for videopath in videopaths:
        parent = videopath.parent
        d = parent.joinpath('frames')
        # if not d.exists:

        f = FFMPEGFrames.FFMPEGFrames(str(d))
        f.extract_frames(input=str(videopath), fps=1)
        
        assert len(list(parent.glob('**/*.png'))) == \
                len(list(parent.glob('**/*.pts')))

if __name__ == '__main__':
    main()