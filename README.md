# Point Cloud Feature extraction based on detections
Part of my master thesis: 
**Probabilistic 3D Multi-Modal Multi-Object Tracking via Machine Learning and Analytic Collision Risk Calculation for Autonomous Vehicles Navigation.**

## Overview
This module is dedicated to extracting 3d detections and their representing point cloud features based on pre-trained centerpoint detectors. A review paper in english is under construction.
**Module is tailored for nuScenes dataset and CenterPoint's trained models.**

## Instructions
### 1. Download and Set Up NuScenes Devkit
Download the [NuScenes Devkit](https://github.com/nutonomy/nuscenes-devkit) and follow the instructions to set it up.
Make sure to add the devkit to your Python path.

### 2. Clone OpenPCDet repo and setup the environment
You should follow the instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git) to [install](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) the repo, [download and prepare the NuScenes dataset info](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) and use the desired [pre-trained model](https://github.com/open-mmlab/OpenPCDet/blob/master/README.md).

### 3. Get Detections and Extract Features
Head to ```pcdet/datasets/nuscenes/nuscenes_dataset.py``` ```L50``` and choose the val or train infos you have created.
Go to the /tools directory.
In ```inference.py``` you must add the destination file and choose an apropriate acceptance detections thresholds.
To get results with voxel size 0.1:
```bash
python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel01_centerpoint_nds_6454.pth
```
Or with voxel 0.075 (suggested):
```bash
python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel0075_centerpoint_nds_6648.pth
```
Change destination paths according to your directory format.


## Acknowledgments
Built on top of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git).
