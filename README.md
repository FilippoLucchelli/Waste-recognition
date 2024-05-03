# Waste-recognition
Repository for training and testing different segmentation models on multispectral waste dataset.

## Models

- UNet
- UNet++
- DeepLabV3
- DeepLabV3+
- ACNet
- MSNet

## Data structure
.
└── root_dir/
    ├── results/
    │   └── model_folder/
    │       ├── model.pth
    │       └── parameters.csv
    └── data_folder/
        └── test/
            ├── band1
            ├── band2
            ├── ...
            └── band n