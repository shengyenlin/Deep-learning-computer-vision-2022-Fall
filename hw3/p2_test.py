import random
import sys
import json
import time 

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tokenizers import Tokenizer
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import torch

from p2.model import make_model
from p2.dataset import P2TestDataset
from p2.preprocess import P2Transformation

# Set random seed for reproducibility
SEED = 5566
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

img_root = sys.argv[1]
out_path = sys.argv[2]
model_path = 'p2.ckpt'
tokenizer_path = 'p2/caption_tokenizer.json'

class Args:
    def __init__(self):
        self.image_size = 224
        self.max_len = 64
        self.batch_size = 1
        self.use_catr = 0
        self.lr_drop=20
        self.encoder_model_name = 'vit_large_patch14_224_clip_laion2b'
        self.decoder_hid_dim = 1024
        self.decoder_n_head = 8
        self.decoder_num_layers = 6
        self.decoder_dropout = 0.1
        self.gen_hid_dim = 1024
        self.gen_num_layers = 2
        self.vocab_size = 18022
        self.layer_norm_eps = 1e-12

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template


@torch.no_grad()
def predict(model, img, caption, cap_mask, args) -> torch.Tensor:
    model.eval()
    for i in range(args.max_len - 1):
        predictions = model(img, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        # [EOS] = 3
        if predicted_id[0] == 3:
            return caption
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
    return caption

def decode_to_text(preds_id, tokenizer):
    preds_text = []
    for pred_id in preds_id:
        pred_text = tokenizer.decode(pred_id)
        preds_text.append(pred_text)
    return preds_text


def save_pred_to_json(imgs_id, preds_all_texts, pred_path):
    pred_dict = dict(zip(imgs_id, preds_all_texts))
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)

def main(args):
    x = time.time()
    transform = Compose([
            Resize((args.image_size, args.image_size)),
            ToTensor()
        ])

    test_ds = P2TestDataset(img_root, transform)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    f = open(tokenizer_path)
    token_json = json.load(f)
    vocab_size = len(token_json['model']['vocab'])

    # load model, model.eval
    model = make_model(args, vocab_size)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    model.eval()
    # predict
    img_names, preds = [], []
    for img_name, img in test_dl:
        # [BOS] = 2
        caption, cap_mask = create_caption_and_mask(
            start_token=2, max_length=args.max_len
            )
        img, caption, cap_mask = img.to(device), caption.to(device), cap_mask.to(device)
        pred_ids = predict(model, img, caption, cap_mask, args)
        pred_ids = pred_ids.cpu().numpy().tolist()
        
        pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        pred = pred.capitalize()
        img_names.append(img_name[0]), preds.append(pred)
        print(img_name[0], pred)
        
    save_pred_to_json(img_names, preds, out_path)

    y = time.time()
    time_elapsed = (y-x)/60
    print(f'testing time: {round(time_elapsed, 3)} mins')

if __name__ == '__main__':
    args = Args()
    main(args)