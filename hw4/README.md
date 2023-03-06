# HW4

Refer to report file `hw4_r11922a05.pdf` for model performance, experiement results, observation and discussions. 

## Problem 1: 3D Novel View Synthesis
- Evaluation metric: PSNR [wiki](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio), SSIM [wiki](https://en.wikipedia.org/wiki/Structural_similarity), LPIPS [arxiv](https://arxiv.org/abs/1801.03924)
- Train my own NeRF [arxiv](https://arxiv.org/abs/2003.08934) model on a one object hotgdog scene.
- To be more specific, given a set of training imag(with camera pose) of this scene, make your model fit this scene. After that, given a test set camera pose, the model should be able to synthesize these unseen novel views of this scene.
- Used Direct Voxel Grid Optimization (DVG) [arxiv](https://arxiv.org/abs/2111.11) to speed up NeRF

## Problem 2: Self-supervised Pre-training for Image Classification 
- Dataset: Mini-ImageNet, Office-Home
- Evaluation metric: Accuracy
- Pre-train my own ResNet50 backbone on Mini-ImageNet via the recently self-supervised learning methods - BYOL [github](https://github.com/lucidrains/byol-pytorch) and Barlow Twins [github](https://github.com/facebookresearch/barlowtwins) [arxiv](https://arxiv.org/abs/2006.0773)

---

## Environment
- python version: 3.8
- packages: pip3 install -r requirements.txt

## Reproduce
```bash
bash data.sh #download data
bash p1_train.bash
bash p2_pretrain.bash
bash p2_finetune.bash
```
## Testing
```bash
bash hw4_download.sh
bash hw4_1.sh [--transform_test_json_path] [--out_img_dir]
bash hw4_2.sh [--img_csv_path] [--img_dir] [--out_csv_path]
```