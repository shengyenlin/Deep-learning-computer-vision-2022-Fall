# HW2

Refer to report file `hw2_r11922a05.pdf` for model performance, experiement results, observation and discussions.

## Problem 1: GAN
- Face dataset: CalebA
- Evaluation metric: Fréchet inception distance [Git](https://github.com/mseitzer/pytorch-fid), face recognition
- Train a DCGAN model from scratch, [Arxiv](https://arxiv.org/abs/1511.06434)
- Improved model: MSGAN (DCGAN + mode seek loss), [Arxiv](https://arxiv.org/abs/1903.05628)

## Problem 2: Diffusion models
- Digit dataset: MNIST-M
- Evaluation metric: Accuracy
- Train a conditional diffusion model with Unet from scratch, [Arxiv](https://arxiv.org/abs/2006.11239)

## Problem 3: DANN
- Digit dataset: MNIST-M, SVHN and USPS
- Evaluation metric: Accuracy
- Train a DANN model with CNN from scratch
- Two scenarios: (a) MNIST-M → SVHN, (b) MNIST-M → USPS 
- Visualize the latent space of DANN by mapping the validation images to 2D space with t-SNE, color data point by class and by domain
---

## Environment
- python version: 3.8.13
- packages: pip3 install -r requirements.txt

## Reproduce
```bash
bash p1_train.bash
bash p2_train.bash
bash p3_train.bash
```
## Testing
```bash
bash hw2_download.sh
bash hw2_1.sh [--output_dir]
bash hw2_2.sh [--output_dir]
bash hw2_3.sh [--svhn_data_dir] [--output_file_path]
```