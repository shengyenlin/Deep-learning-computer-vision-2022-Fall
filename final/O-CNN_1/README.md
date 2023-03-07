## Running resources
NVIDIA A100, Driver Version: 515.65.01, CUDA Version: 11.7
 

1. Enter the subfolder `pytorch`, and install PyTorch and relevant packages with
   the following commands:
    ```shell
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
    <!--
    - `pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1` conda  failed
    - `pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel`        docker failed
    - `pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel`        docker succeed
    -->

2. Build O-CNN under PyTorch.
   ```shell
   python setup.py install --build_octree
   ```

3. Run the test cases.
   ```shell
   python -W ignore test/test_all.py -v
   ```

## Download checkpoint
1. Enter the subfolder `pytorch` and run `final_download.sh` by: bash final_download.sh


## Data preparation
1. Put `train`, `test`, data folder under `O-CNN_1/pytorch/projects/data/scannet/` 
2. Run `python tools/scannet.py --run process_scannet --path_in <input_folder_path>`, where `<input_folder_path>` should be `data/scannet/` if following step 1.

## Run for experiment training
1. Run `python segmentation.py --config configs/seg_scannet.yaml`

## Run inference code for output folder
1. Run `python segmentation.py --config configs/seg_scannet_eval.yaml`

## Run for predict txt file
1. Run `python tools/scannet.py --path_in data/scannet/test --path_pred logs/scannet/v1000_eval --path_out <output_folder_path> --filelist data/scannet/scannetv2_test_new.txt --run generate_output_seg`


## Zip the file
1. Run `python logs/scannet/run_zip.py` to get `ocnn_1000.zip` under path `O-CNN_1/pytorch/projects/`
  

