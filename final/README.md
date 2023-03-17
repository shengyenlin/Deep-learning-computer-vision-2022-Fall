# DLCV final project - 3D Indoor Scene Long Tail Segmentation

## Competition description

The goal of this competition is to train a neural network to accurately classify the semantic labels of each point in a given 3D point cloud scene. The input for this competition will consist of a 3D point cloud scene, which includes the XYZ position and RGB color (optional) for each point. The output that you will be tasked with generating is a semantic class label for each point in the point cloud.

## Competition result

Link to competition: [Codalab](https://codalab.lisn.upsaclay.fr/competitions/8961?secret_key=0865b2c6-96da-4725-86a5-dd793d)

Leaderboard: 5/15
Oral presentation: 2/15  

## Guildline

To reproduce our result, please generate three predictions with OCNN, OCNN+pretraining and OCNN+mix3d, and use voting to get the final result.

## Download checkpoints
Run `bash final_download.sh` to download checkpoints

## 1. Original OCNN

### Running resources
NVIDIA A100, Driver Version: 515.65.01, CUDA Version: 11.7

### Installation (hard to install)
1. Enter the subfolder `O-CNN_1/pytorch`, and install PyTorch and relevant packages with
   the following commands:


```python
    conda create --name pytorch-1.9.0 python=3.7
    conda activate pytorch-1.9.0
    conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
    pip install -r requirements.txt
    pip uninstall setuptools
    pip install setuptools==59.5.0
    pip install pytorch-lightning
```

The code is also tested with the following pytorch version:
- `pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1`
- `pytorch==1.9.0 torchvision cudatoolkit=11.1 `

2. Build O-CNN under PyTorch.


```python
python setup.py install --build_octree
```

3. Run the test cases.


```python
python -W ignore test/test_all.py -v
```

### Data preparation
1. Change the directory into `O-CNN_1/pytorch/projects` 
2. Run `python tools/scannet.py --run process_scannet --path_in <input_folder_path>`

### Run for experiment training
Run `python segmentation.py --config configs/seg_scannet.yaml`

### Run inference code for output folder
Run `python segmentation.py --config configs/seg_scannet_eval.yaml`

### Run for predict txt file
Run `python tools/scannet.py --path_in data/scannet/test --path_pred logs/scannet/v1000_eval --path_out logs/scannet/v1000_eval_seg --filelist data/scannet/scannetv2_test_new.txt --run generate_output_seg`


### Zip the file
Run `python logs/scannet/run_zip.py` to get `ocnn_1000.zip` under path `O-CNN_1/pytorch/projects/`

## 2. OCNN with pretraining & OCNN with Mix3D

### Installation (easier to install)
1. Create a virtual environment with python version>=3.7 and activate the environment.
2. Install Pytorch. The code has been tested with Pytorch>=1.6.0, and Pytorch>=1.9.0 is preferred.
3. Enter the subfolder `O_CNN_2/projects`. Run `pip install -r requirements.txt`

### Data preparation
Run `python tools/seg_scannet.py --run process_scannet --path_in <input_folder_path>`

### Run for experiment training
Run `python segmentation.py --config configs/seg_scannet.yaml`

### Run inference code for output folder
1. For OCNN with Mix3D, run `python segmentation.py --config configs/seg_scannet_eval.yaml`

2. For OCNN with pretraining, run `python segmentation.py --config configs/seg_scannet_eval_1.yaml`

### Run for predict txt file

1. For OCNN with Mix3D, run `python tools/seg_scannet.py --path_in data/scannet/test --path_pred logs/scannet/v400_eval --path_out logs/scannet/v400_eval_seg --filelist data/scannet/scannetv2_test_new.txt --run generate_output_seg`

2. For OCNN with pretraining, run `python tools/seg_scannet.py --path_in data/scannet/test --path_pred logs/scannet/v300_eval --path_out logs/scannet/v300_eval_seg --filelist data/scannet/scannetv2_test_new.txt --run generate_output_seg`

### Zip the file
1. For Mix3D setting, run `python logs/scannet/run_zip_mix.py` to get`ocnn_400.zip` under path `O_CNN_2/projects/` 

2. For contrastive pretraining setting, run `python logs/scannet/run_zip_pre.py` to get `ocnn_300.zip` under path `O_CNN_2/projects/`

## Voting
Go back to the root. Use majority vote to get the final output.

```python
python vote.py --inp O-CNN_1/pytorch/projects/ocnn_1000.zip O_CNN_2/projects/ocnn_400.zip O_CNN_2/projects/ocnn_300.zip --dest vote.zip
```
