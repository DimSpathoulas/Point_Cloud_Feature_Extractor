# Point Cloud Feature extraction based on detections
Part of my master thesis: 
**Probabilistic 3D Multi-Modal Multi-Object Tracking Using Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**

## Overview
This module is dedicated to extracting 3d detections and their representing point cloud features based on a pre-trained centerpoint detector.

CenterPoint detects objects. The detected centers of objects are inversely transformed from the lidar frame to the BEV feature map. We then extract a point cloud feature. This point cloud feature consists of a 3×3 patch around the center and its respected 512 vector from the feature map.

pcet/datasets/nuscenes/nuscenes_dataset.py 38, 44, 119

pcdet/models/dense_heads/center_head.py

tools/inference.py
