import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pic_dir",
        type=Path,
        help="Directory to the valid picture dataset.",
        default="./hw3_data/p1_data/val",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default="./cache/p1"
    )
    parser.add_argument(
        "--id2label_path",
        type=Path,
        default='./hw3_data/p1_data/id2label.json'
    )
    parser.add_argument("--mode", type=str, help="val, test", default="test")
    # model
    parser.add_argument("--model", type=str, help="RN50, RN101, RN50x4, RN50x16, ViT-B/32, ViT-B/16", default="ViT-B/16")
    parser.add_argument("--prompt", type=int, default=1)
    parser.add_argument("--n_top", type=int, help="number of top indices chosen", default=1)
    parser.add_argument("--n_chunk", type=int, default=100, help = "number of chunks to use in evaluation to avoid OOM")
    args = parser.parse_args()
    return args

def read_img(val_imgs_path, preprocess, args):
    imgs_preprocessed = []
    imgs_original = []
    for val_img in val_imgs_path:
        path = os.path.join(args.pic_dir, val_img)
        img = Image.open(path)
        img = img.convert("RGB")
        imgs_original.append(img)
        img = preprocess(img)
        imgs_preprocessed.append(img)
    imgs_ts = torch.tensor(np.stack(imgs_preprocessed))
    return imgs_ts, imgs_original

def read_text(num_prompt, id2label):
    texts = []
    for label in id2label.values():
        if num_prompt == 1:
            prompt = f"This is a photo of a {label}."
        elif num_prompt == 2:
            prompt = f"This is a {label} image."
        elif num_prompt == 3:
            prompt = f"No {label}, no score."
        elif num_prompt == 4:
            prompt = f"a photo of a {label}"
        texts.append(clip.tokenize(prompt))
    texts_ts = torch.cat(texts)
    return texts_ts


def compute_sim(model, imgs, texts, n_top):
    img_features, text_features = get_feature(model, imgs, texts)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)    
    similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity.topk(n_top, dim=-1)
    return values, indices.cpu().numpy()

def get_feature(model, imgs, texts):
    with torch.no_grad():
        img_features = model.encode_image(imgs)
        text_features = model.encode_text(texts)
    return img_features, text_features

def main(args):
    with open(args.id2label_path) as json_file:
        id2label = json.load(json_file)

    val_imgs_path = sorted(os.listdir(args.pic_dir))
    
    model, preprocess = clip.load(args.model, device)
    imgs, _ = read_img(val_imgs_path, preprocess, args) # (num_img, c, h, w)
    imgs = imgs.to(device)
    texts = read_text(args.prompt, id2label).to(device) # (num_class, 77)
    
    indices = []
    model.eval()
    for img_chunk in imgs.chunk(args.n_chunk):
        values, index_chunk = compute_sim(model, img_chunk, texts, args.n_top)
        index_chunk = index_chunk.flatten()
        indices.append(index_chunk)
    
    indices = np.concatenate(indices, axis=0)

    if args.mode == 'val':
        gt_ids = [int(path.split("_")[0]) for path in val_imgs_path]
        correct = np.sum(gt_ids == indices)
        acc =  correct / len(gt_ids)
        print(f"Accuracy = {acc}")
    
    df = pd.DataFrame(
        {
            "filename": val_imgs_path,
            "label": indices
        }
    )
    df.to_csv(args.out_dir, index=False) 
    

if __name__ == '__main__':
    args = parse_args()
    # args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args)