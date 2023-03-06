# HW3

Refer to report file `hw3_r11922a05.pdf` for model performance, experiement results, observation and discussions.

## Problem 1: Zero-shot image classification with CLIP
- Evaluation metric: Accuracy
- Evaluate the pretrained CLIP on image classification task
- Compare CLIP with VGG and Resnet, explain why CLIP could achieve competitive zero-shot performance on a great variety of image classification datasets
- Compare and discuss the performances of your model with different prompts

## Problem 2: Image Captioning with Vision and Language Model
- Evaluation metric: CIDEr [Arxiv](https://arxiv.org/pdf/1411.5726.pdf) [Toolkit](https://github.com/bckim92/language-evaluation), CLIPScore [Arxiv](https://arxiv.org/pdf/2104.08718)
- Implement a Vision and Language (VL) model for image captioning
  - Vision: ViT
  - Language: Transformer decoder [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)
- Implement a quantitative metric, namely CLIPScore, for evaluation.
  
## Problem 3: Visualization of Attention in Image Captioning
- Analyze the image captioning model in problem 2 by visualizing the cross-attention between images and generated captions
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
bash get_dataset.sh
bash hw3_download.sh
bash hw3_1.sh [--test_img_dir] id2label.json [--out_csv_path]
bash hw3_2.sh [--test_img_dir] [--out_json_path]
```