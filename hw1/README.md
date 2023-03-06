# HW1

## Problem 1: Image classificaion
- Train a CNN model from scratch
- Use ResNet152 to improve the previous model
- Visualize learned visual representation of second last layer by PCA and t-SNE

## Problem 2: Semantic segmentation
- Implement VGG16 + FCN32s (baseline model)
- Improved model: Segformer-b1-finetuned-ade-512-512, [hugging face](https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512)

---

## Environment
- python version: 3.8.13
- packages: pip3 install -r requirements.txt

## Reproduce
```bash
python3 p1_a_train.py
python3 p1_b_train_cv.py
python3 p1_b_train.py
python3 p2_a_train.py
python3 p2_b_segformer_train.py
```
## Testing
```bash
bash hw1_download.sh
bash hw1_1.sh [--img_dir] [--output_csv_path]
bash hw1_2.sh [--img_dir] [--output_dir]
```

## Check test result on valid set
- p1: python3 p1_recheck.py [--img_dir] [--output_csv_path]
- p2: python3 p2_recheck.py [--img_dir] [--output_dir]
