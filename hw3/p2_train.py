import random
import time 
import datetime
import os
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
import gc

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tokenizers import Tokenizer
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import torch
import torch.nn as nn
import torch.optim as optim

from p2.dataset import P2Dataset, pad_collate
from p2.model import make_model
from p2.preprocess import P2Transformation

# Set random seed for reproducibility
SEED = 5566
print("Random Seed:", SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
torch.set_num_threads(1)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_img_dir",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw3_data/p2_data/images/train"
    )
    parser.add_argument(
        "--val_img_dir",
        type=Path,
        help="Directory to the valid dataset.",
        default="./hw3_data/p2_data/images/val",
    )

    parser.add_argument(
        "--train_text_path",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw3_data/p2_data/train_organized.json",
    )
    parser.add_argument(
        "--val_text_path",
        type=Path,
        help="Directory to the valid dataset.",
        default="./hw3_data/p2_data/val_organized.json",
    )

    parser.add_argument(
        "--val_text_orign_path",
        type=Path,
        help="Directory to the original valid text dataset.",
        default="./hw3_data/p2_data/val.json"
    )

    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/p2/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/p2/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="path to pretrained tokenizer",
        default="./p2/caption_tokenizer.json",
    )

    parser.add_argument("--use_catr", type=int, default=0)

    # optimizer
    parser.add_argument("--optimizer", type=str, default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_drop", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # img
    parser.add_argument("--image_size", type=int, default=224)

    # encoder
    parser.add_argument("--encoder_model_name", type=str, default='vit_base_patch16_224')
    parser.add_argument("--freeze_encoder", type=int, default=0)

    # decoder
    parser.add_argument("--decoder_hid_dim", type=int, default=768, help='the number of expected features in the input of decoder')
    parser.add_argument("--decoder_n_head", type=int, default=8, help='number of head in multi-head self attention')
    parser.add_argument("--decoder_num_layers", type=int, default=12, help = 'the number of sub-decoder-layers in the decoder')
    parser.add_argument("--decoder_dropout", type=float, default=0.1, help='dropout rate in positional encoding')
    parser.add_argument("--vocab_size", type=int, default=18022)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    # generator
    parser.add_argument("--gen_hid_dim", type=int, default=1024, help='hidden dim of each layer in MLP')
    parser.add_argument("--gen_num_layers", type=int, default=2, help='number of layer in MLP')
    parser.add_argument("--max_len", type=int, default=64)

    # training 
    parser.add_argument("--device", type=torch.device, help="cpu, cuda", default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--eval_num_per_iter", type=int, default=100)
    parser.add_argument("--mode", type=str, help='val test', default='val')

    parser.add_argument("--clip_max_norm", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=36)
    parser.add_argument("--val_batch_size", type=int, default=10)
    parser.add_argument("--dynamic_pad", type=bool, default=False)

    args = parser.parse_args()
    return args

def update_best_metric(hist, best):
    best['epoch'] = hist['epoch'][-1]
    # best['iter'] = hist['iter'][-1]
    best['train_loss'] = hist['train_loss'][-1]
    best['val_loss'] = hist['val_loss'][-1]

def save_model(model, n_epoch, model_save_path):
    path = os.path.join(model_save_path, f'ep_{n_epoch}.ckpt')
    torch.save({'model': model.state_dict()}, path)

def save_best(model, best, hist, model_save_path):
    if hist['val_loss'][-1] <= best['val_loss']:
            update_best_metric(hist, best)

            filename = 'ep_%d_loss_%.4f' % (best['epoch'], best['val_loss'])
            path = os.path.join(model_save_path, filename)
            torch.save(
                {
                    'model': model.state_dict(),
                }, path
            )

            best_path = os.path.join(model_save_path, 'best.ckpt')
            try:
                Path(best_path).unlink()
            except:
                pass
            Path(best_path).symlink_to(filename)

def save_best_to_txt(cache_dir, best, args_dict):
    with open(Path(cache_dir) / 'best.txt', 'w') as f:
        print(best, file=f)
        print(args_dict, file=f)
    f.close()    

def init_weight(model):
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    model_dict = model.state_dict()
    layers = [k for k in model_dict.keys() if 'encoder' not in k]
    for layer in layers:
        model_dict[layer] = torch.randn(model_dict[layer].shape) * 0.01
    # for names, params in model.named_parameters():
    #     # encoder, decoder + decoder embedding + MLP
    #     if 'encoder' not in names:
    #         if params.dim() > 1:
    #             # nn.init.xavier_uniform_(params)
    #             nn.init.normal_(params)
    return model

def train(dl, model, optimizer, loss_fn, hist, max_norm):
    train_loss = 0
    num_batch = 0
    for data in tqdm(dl, leave=False, colour='green'):
        img, id, id_y, attn_mask = data['img'], data['cap_id'], data['cap_id_y'], data['cap_attn_mask']
        img, id, id_y, attn_mask = img.to(device), id.to(device), id_y.to(device), attn_mask.to(device)
        model.zero_grad()
        #(bs, seq_len, token_size)
        prob = model(img, id, attn_mask)
        
        # nn.CrossEntropy(input, target)
        # input = (N, C, d1, d2, ..., dk), target (gt) = (N, d1, d2, ..., dk)
        loss = loss_fn(prob.permute(0, 2, 1), id_y)
        loss.backward()
        train_loss += loss.item()
        

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        _, pred = torch.max(prob, dim=2)
        # print("input:", id)
        # print("gt:", id_y[:10])
        # print("pred:", pred[:10])
        # print(loss)
        del img, id, id_y, attn_mask, prob, loss
        num_batch += 1
        gc.collect()
        torch.cuda.empty_cache()
        
    train_loss = train_loss / num_batch
    hist['train_loss'].append(train_loss)

def create_caption_and_mask(bs, start_token, max_length):
    caption_template = torch.zeros((bs, max_length), dtype=torch.long)
    mask_template = torch.ones((bs, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def greedy_decode(model, img, max_len, start_token, vocab_size):
    model.eval()
    bs = img.size(0)
    memory, pos = model.encode(img)
    preds_id = torch.zeros(bs, max_len, dtype=torch.long).fill_(start_token).to(device)
    preds_prob = torch.zeros(bs, max_len, vocab_size).to(device)
    for i in range(max_len):
        attn_mask = torch.zeros(preds_id.size()).bool().to(device)
        decode_out = model.decode(
            memory,
            pos,
            preds_id, 
            tgt_pad_mask=attn_mask #need 0000
        )
        prob = model.generator(decode_out) # (nb, seq_len, vocab_size)

        _, pred_sentences = torch.max(prob, dim=2) # (nb, seq_len)
        next_word_prob = prob[:, -1, :]
        next_word = pred_sentences[:, -1]
        
        preds_id = torch.cat(
            #[preds_id, id_y[:, i].unsqueeze(1)], dim=1
            [preds_id, next_word.unsqueeze(1)], dim=1
        )
        print(preds_id)
        preds_prob[:, i, :] = next_word_prob  # (nb, 1, vocab_size)
    # (nb, test_max_len, vocab_size), (nb, test_max_len, vocab_size)
    return preds_id, preds_prob

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

@torch.no_grad()
def eval(val_dl, model, loss_fn, hist):
    model.eval()
    val_loss = 0
    num_batch = 0 
    for data in tqdm(val_dl, leave=False, colour='green'):
        img, id, id_y, attn_mask = data['img'], data['cap_id'], data['cap_id_y'], data['cap_attn_mask']
        img, id, id_y, attn_mask = img.to(device), id.to(device), id_y.to(device), attn_mask.to(device)

        prob = model(img, id, attn_mask)
        loss = loss_fn(prob.permute(0, 2, 1), id_y)
        val_loss += loss.item()
        del img, id, id_y, attn_mask, prob, loss
        gc.collect()
        torch.cuda.empty_cache()
        num_batch += 1
        
    val_loss = val_loss / num_batch
    hist['val_loss'].append(val_loss)

def freeze_encoder(model):
    for names, params in model.named_parameters():
        if 'encoder' in names:
            params.requires_grad = False
    return model

def plot_loss(hist, cache_dir):
    train_loss = hist['train_loss']
    val_loss = hist['val_loss']
    len_train = len(train_loss)
    len_val = len(val_loss)
    x_train = np.arange(1, len_train+1, 1)
    x_val = np.arange(1, len_val+1, 1) 

    plt.plot(x_train, train_loss, label = 'train loss')
    plt.plot(x_val, val_loss, label = 'valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss plot')
    plt.legend()
    plt.savefig(
        os.path.join(cache_dir, 'loss plot')
    )
    plt.close()

def main(args):
    run_id = int(time.time())
    args_dict = vars(args)
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")

    cache_dir = args.cache_dir / str(date) / str(run_id)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    f = open(args.tokenizer_path)
    token_json = json.load(f)
    vocab_size = len(token_json['model']['vocab'])
    
    # dataset: img, img_mask, caption, caption_mask
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
        ])
    p2_trans = P2Transformation()
    #p2_trans.transform['train']
    train_ds = P2Dataset(
        args.train_img_dir, args.train_text_path, tokenizer, 
        args.max_len, transform=p2_trans.transform['train'], dy_pad=args.dynamic_pad
        )
    val_ds = P2Dataset(
        args.val_img_dir, args.val_text_path, tokenizer, 
        args.max_len, transform=p2_trans.transform['val'], dy_pad=args.dynamic_pad
        )
    
    if args.dynamic_pad:
        train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=pad_collate
        )

        val_dl = torch.utils.data.DataLoader(
            val_ds, 
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=pad_collate
        )
    else:
        train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        val_dl = torch.utils.data.DataLoader(
            val_ds, 
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
    if args.use_catr:
        model = make_model(args, vocab_size)
        if args.freeze_encoder:
            model = freeze_encoder(model)
    else:
        model = make_model(args, vocab_size)
        if args.freeze_encoder:
            model = freeze_encoder(model)

    model = init_weight(model)
    model = model.to(device)
    print(model.decoder)
    # print(model.state_dict())

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr
        )
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr
        )   
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, 
            weight_decay=args.weight_decay
        )       

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop
        )

    hist = {
        'epoch': [],
        # 'iter': [],
        'train_loss': [],
        'val_loss': []
    }

    best = {
        'epoch': 0,
        # 'iter': 0,
        'train_loss': np.inf,
        'val_loss': np.inf
    }

    # set up computer


    # train
    iters = 0
    patience = 5
    not_update = 0
    x = time.time()
    for epoch in range(1, args.num_epoch+1):
        hist['epoch'].append(epoch)
        train(train_dl, model, optimizer, loss_fn, hist, args.clip_max_norm)
        lr_scheduler.step()   
        eval(val_dl, model, loss_fn, hist)

        plot_loss(hist, cache_dir)
        if hist['val_loss'][-1] > best['val_loss']:
            not_update += 1

        #save_best(model, best, hist, cache_dir)
        save_model(model, epoch, cache_dir)
        gc.collect()
        torch.cuda.empty_cache()
        y = time.time()
        time_passed = (y-x) / 60

        print('[%d/%d] \ttrain loss: %.4f\tvalid loss: %.4f\t%.3f mins elapsed'
            % (epoch, args.num_epoch,
                hist['train_loss'][-1], hist['val_loss'][-1], round(time_passed)),
                end = '\n'
        )

        if not_update == patience:
            print("Early stopping")
            print("Best result:")
            print('Epoch: %d \ttrain loss: %.4f\tvalid loss: %.4f'
            % (best['epoch'],best['train_loss'], best['val_loss']),
                end = '\n'
            )


if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)