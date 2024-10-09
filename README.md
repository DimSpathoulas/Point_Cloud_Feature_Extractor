# UNDER CONSTRUCTION #

This repository will not work properly as it misses important that will be added after thesis defense ####
# Point Cloud Feature extraction based on detections
Part of my master thesis: 
**Probabilistic 3D Multi-Modal Multi-Object Tracking Using Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**

## Overview
This module is dedicated to extracting 3d detections and their representing point cloud features based on a pre-trained centerpoint detector.

CenterPoint detects objects. The detected centers of objects are inversely transformed from the lidar frame to the BEV feature map. 

We then extract a point cloud feature. This point cloud feature consists of a 3×3 patch around the center. Feature map has 512 layers depth.

**Module is tailored for nuScenes dataset and CenterPoint's trained model.**

pcet/datasets/nuscenes/nuscenes_dataset.py 38, 44, 119

pcdet/models/dense_heads/center_head.py

tools/inference.py


## TODO 
1. Add instructions on how to run the script on the README.  
1.1 Which environment to create (and how), activate it, and then run whatever is needed.  
1.2 What weights are needed, where to find them and where to place them.


2. Probably to run, you need:   
`cd tools;`
`python inference.py --cfg_file cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml --data_path /second_ext4/ktsiakas/kosmas/nuscenes --ckpt ../cbgs_voxel01_centerpoint_nds_6454.pth`  
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

Currently:  
- screen #1: openpcdet, running for the val set
- screen #1: openpcdet_train, running for the train set

L94 in Inference actually initializes the mode train for centerpoint detector aswell...
Dont know how to add an input train or val for the train split val explicitly...
nuscenes_dataset.py L20 : THE INPUT train (from nusc train split) actually initializes state train for centerpoint aswell
and L50 is the actual input
