# Description

This is the code for my thesis titled "Application of Different Similarity Measures in Spectral Clustering for Individual Tree Detection and Segmentation in LiDAR Data". It is written to work with the NEWFOR dataset [1]. This repository does not contain the dataset, so you need to download it from its [official site](https://www.newfor.net/download-newfor-single-tree-detection-benchmark-dataset/).

This code uses the spectral clustering algorithm with Nyström approximation from [2]. Its original code, slightly modified to fit our interface, is situated in `nsc/original.py`. The original code was used as a reference for the implementation of our method.

Experiments are run using the master script `master.py`. You need to set constants `DATASET_PATH` and `WORKING_PATH` to a folder with NEWFOR benchmark data and a set of empty folders with the same structure, respectively.

The code used to create plots is situated in `plots.py`. It is not run as a part of the experiments.

## Dependecies

* numpy
* matplotlib
* scikit-learn
* fiona
* laspy
* rasterio
* shapely

## References

1. Eysn, Lothar, et al. "A benchmark of lidar-based single tree detection methods using heterogeneous forest data from the alpine space." Forests 6.5 (2015): 1721-1747.
2. Pang, Yong, et al. "Nyström-based spectral clustering using airborne LiDAR point cloud data for individual tree segmentation." International Journal of Digital Earth 14.10 (2021): 1452-1476.
