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

    # Define the path to the output JSON file
    # if args.train:
    #     output_file = "centerpoint_predictions_train.npy" 
    # else:
    #     output_file = "centerpoint_predictions_val_2.npy" 

    output_file = "centerpoint_predictions_train_2.npy" 

    if os.path.exists(output_file):
        os.remove(output_file)

    all_pred_dicts = []
    sample_count = 0
    save_interval = 400
    print('Starting evaluation')
    
    with torch.no_grad():
        for idx, data_dict in enumerate(nusc_dataset):
            # print(idx)
            torch.cuda.empty_cache()    
            # print(f"Iteration {idx + 1} / n")
            data_dict = nusc_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_dict = pred_dicts[0]

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

            new_dict = {'metadata': pred_dict['metadata']}
            # new_dict = {'metadata': pred_dict['metadata'].tolist()}

            threshold_mask = pred_dict['pred_scores'] > 0.03  # kapoy 0.61 eida einai ok.... mporei kai ligo pio kato

            for key, value in pred_dict.items():
                if key != 'metadata':
                    pred_dict[key] = value.cpu().numpy()
                    new_dict[key] = value[threshold_mask]
                    # if isinstance(value, torch.Tensor):

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=new_dict['pred_boxes'],
            #     ref_scores=new_dict['pred_scores'], ref_labels=new_dict['pred_labels']
            # )

            # print(new_dict)
            # print(new_dict['metadata'])

            all_pred_dicts.append(new_dict)
            sample_count += 1

            # print('all preds len', len(all_pred_dicts))

            if sample_count % save_interval == 0:
                save_and_stack_results(all_pred_dicts, output_file)
                all_pred_dicts = []  # Clear the list after saving

            torch.cuda.empty_cache()

        print('complete len counter', sample_count)
        torch.cuda.empty_cache()
        if all_pred_dicts:
            save_and_stack_results(all_pred_dicts, output_file)
            # print('final all preds len', len(all_pred_dicts))

        print("Saved predictions")


if __name__ == '__main__':
    main()
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


def concatenate_all_files(base_output_file, num_files, final_output_file):
    all_results = []

    for i in range(num_files):
        output_file = f"{base_output_file}_{i}.npy"
        if os.path.exists(output_file):
            results = np.load(output_file, allow_pickle=True)
            all_results.append(results)

    # Concatenate all results into one
    combined_results = np.concatenate(all_results)
    
    # Save the final concatenated results
    np.save(final_output_file, combined_results)
    print(f"Concatenated all {num_files} files into {final_output_file}")


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

    # Define the path to the output JSON file
    # if args.train:
    #     output_file = "centerpoint_predictions_train.npy" 
    # else:
    #     output_file = "centerpoint_predictions_val_2.npy" 

    output_file = "centerpoint_predictions_train_2.npy" 

    if os.path.exists(output_file):
        os.remove(output_file)

    all_pred_dicts = []
    sample_count = 0
    save_interval = 400
    print('Starting evaluation')
    
    with torch.no_grad():
        for idx, data_dict in enumerate(nusc_dataset):
            # print(idx)
            torch.cuda.empty_cache()    
            # print(f"Iteration {idx + 1} / n")
            data_dict = nusc_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_dict = pred_dicts[0]

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

            new_dict = {'metadata': pred_dict['metadata']}
            # new_dict = {'metadata': pred_dict['metadata'].tolist()}

            threshold_mask = pred_dict['pred_scores'] > 0.03  # kapoy 0.61 eida einai ok.... mporei kai ligo pio kato

            for key, value in pred_dict.items():
                if key != 'metadata':
                    pred_dict[key] = value.cpu().numpy()
                    new_dict[key] = value[threshold_mask]
                    # if isinstance(value, torch.Tensor):

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=new_dict['pred_boxes'],
            #     ref_scores=new_dict['pred_scores'], ref_labels=new_dict['pred_labels']
            # )

            # print(new_dict)
            # print(new_dict['metadata'])

            all_pred_dicts.append(new_dict)
            sample_count += 1

            # print('all preds len', len(all_pred_dicts))

            if sample_count % save_interval == 0:
                save_and_stack_results(all_pred_dicts, output_file)
                all_pred_dicts = []  # Clear the list after saving

            torch.cuda.empty_cache()

        print('complete len counter', sample_count)
        torch.cuda.empty_cache()
        if all_pred_dicts:
            save_and_stack_results(all_pred_dicts, output_file)
            # print('final all preds len', len(all_pred_dicts))

        print("Saved predictions")


if __name__ == '__main__':
    main()
