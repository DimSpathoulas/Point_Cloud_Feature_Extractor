# Point Cloud Feature extraction based on detections
Part of my master thesis: 
**Probabilistic 3D Multi-Modal Multi-Object Tracking Using Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**

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
Head to ```pcdet/datasets/nuscenes/nuscenes_dataset.py``` and choose the val or train infos you have created.
Go to the /tools directory.
In ```inference.py``` you must add the destination file and choose an apropriate acceptance detections thresholds.
To get results with voxel size 0.1:
```bash
python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel01_centerpoint_nds_6454.pth
```
While with voxel 0.075 (suggested):
```bash
python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel0075_centerpoint_nds_6648.pth
```
Change destination paths according to your directory format.



pcet/datasets/nuscenes/nuscenes_dataset.py 38, 44, 119

pcdet/models/dense_heads/center_head.py

tools/inference.py


## TODO 
1. Add instructions on how to run the script on the README.  
1.1 Which environment to create (and how), activate it, and then run whatever is needed.  
1.2 What weights are needed, where to find them and where to place them.


2. Probably to run, you need:   
`cd tools;`
`CUDA_VISIBLE_DEVICES=0 python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel01_centerpoint_nds_6454.pth`  
or
`python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel0075_centerpoint_nds_6648.pth`  

or (orevall better performance allegedly)
nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py with weights https://github.com/tianweiy/CenterPoint/blob/master/configs/nusc/README.md



change line in pcdet/datasets/nuscenes/nuscenes_dataset.py L49 OR L50

FROM CFGS nuscenes_yaml DATA_PATH: '/second_ext4/ktsiakas/kosmas/nuscenes'
3. inference.py L104: Make the output also an argument – You cant be sure that this directory exists for everyone, the setup with data, make it an argument. Also customize it if it is train or val

4. [DONE] inference.py L111 comment this thing out: if idx > 40:  break

5. [DONE] Make the script accept as an argument if it is train or val, don’t hardcode this. So, this argument will also go into the datasets/nuscenes/nuscenes_dataset.py L38

6. The change in models/dense_heads/center_head.py, WHY? What does it change from the original one?

7. [DONE] Nuscenes_dataset.py L123, why this? Remove all the hard-coded things. 
