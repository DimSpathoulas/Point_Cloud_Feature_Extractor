import argparse
import glob
from pathlib import Path
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import json

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu

from pcdet.datasets.nuscenes import nuscenes_dataset
from nuscenes.nuscenes import NuScenes
from pcdet.utils import common_utils



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        # print(root_path)
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError


        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def save_and_stack_results(all_pred_dicts, output_file):

    if os.path.exists(output_file):
        # Load existing results
        existing_results = np.load(output_file, allow_pickle=True)
        # Stack new results with existing ones
        combined_results = np.concatenate((existing_results, all_pred_dicts))

    else:
        combined_results = np.array(all_pred_dicts)
    
    # Save combined results
    np.save(output_file, combined_results)
    print(f"Saved {len(combined_results)} samples to {output_file}")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument("--train", action="store_true", 
                    help="use train set or val") 

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def process_and_save_predictions(model, nusc_dataset, output_file, start_idx, end_idx):
    if os.path.exists(output_file):
        os.remove(output_file)

    all_pred_dicts = []
    sample_count = 0
    save_interval = 500
    print(f'Starting evaluation for samples {start_idx} to {end_idx}')

    with torch.no_grad():
        for idx in range(start_idx, end_idx):
            data_dict = nusc_dataset[idx]
            torch.cuda.empty_cache()
            data_dict = nusc_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_dict = pred_dicts[0]

            new_dict = {'metadata': pred_dict['metadata']}
            threshold_mask = pred_dict['pred_scores'] > 0.25

            for key, value in pred_dict.items():
                if key != 'metadata':
                    pred_dict[key] = value.cpu().numpy()
                    new_dict[key] = value[threshold_mask]

            all_pred_dicts.append(new_dict)
            sample_count += 1

            if sample_count % save_interval == 0:
                save_and_stack_results(all_pred_dicts, output_file)
                all_pred_dicts = []  # Clear the list after saving

            torch.cuda.empty_cache()

    print('complete len counter', sample_count)
    torch.cuda.empty_cache()
    if all_pred_dicts:
        save_and_stack_results(all_pred_dicts, output_file)

    print(f"Saved predictions to {output_file}")



def main():
    args, cfg = parse_config()

    logger = common_utils.create_logger()

    nusc_dataset = nuscenes_dataset.NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=nusc_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    samples_train = len(nusc_dataset)  # Get the actual number of samples in the dataset
    samples_per_file = 3600
    num_files = (samples_train + samples_per_file - 1) // samples_per_file  # Calculate the number of files needed

    for i in range(num_files):
        output_file = f"multiple_files/centerpoint_predictions_train_025_{i+1}.npy"
        start_idx = i * samples_per_file
        end_idx = min(start_idx + samples_per_file, samples_train)
        
        process_and_save_predictions(model, nusc_dataset, output_file, start_idx, end_idx)

if __name__ == '__main__':
    main()