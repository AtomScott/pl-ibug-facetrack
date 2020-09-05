# Pytorch-Lightning Data Module for Face Tracking

This repository contains code that can be used to create a pytorch lightning data module from the following datasets:

- 300 Videos in the Wild ([300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/))
- The CONFER Database ([CONFER](https://ibug.doc.ic.ac.uk/resources/confer/))

Boundary boxes for face tracking can be made by using the min/max of the x and y coordinates of the facial landmarks as can be seen here:

![Image](./images/minmax.png)

After running `300vw_preprocess.py` and `confer_preprocess.py`, you will have several sets of images and annotations.
Examples of how a boundary box can be visualized is contained in notebooks/visualize_results.ipynb.

The resulting directory structure should be as follows:

```
data/300VW/                                                   
├── 001
│   ├── annot
│   ├── framesvid
│   └── vid.avi
├── 002
│   ├── annot
...

data/CONFER/
├── FOLD_1
│   ├── 20120305_seq1
│   │   ├── 20120305_seq1_01_01
│   │   ├── 20120305_seq1_01_02
│   │   └── stitched
│   ├── 20120604_seq1
│   │   ├── 20120604_seq1_01_01
...
```

A sample is shown below:

![Example 1.](./images/ex1.png)

![Example 2.](./images/ex2.png)


Pass into the data_module path to folders that contain the .pts files and .png/.jpg of a single face track.
If your directory structure is the same as that of above then the paths will be of the structure,

- data/300VW/001/ (glob: 'data/300VW/**')
- data/CONFER/FOLD_1/20120305_seq1/stitched (glob: 'data/CONFER/**/stitched')

use the glob to gather all the folders or just run `python data_module.py`. `data_module.main()` will do a quick validation run of the gathered files.
