import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models

from p2.dataset import OfficeDataset
from p2.utils import get_office_tarnsform

# Set random seed for reproducibility
SEED = 5566
print("Random Seed:", SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--in_csv_path",
        type=Path
    )
    parser.add_argument(
        "--out_csv_path",
        type=Path
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the training dataset."
    )

    parser.add_argument(
        "--model_path",
        default='./p2_model'
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    return args

def main(args):

    transform = get_office_tarnsform(mode='valid')
    
    # load train dataset
    test_ds = OfficeDataset(
        img_root=args.data_dir,
        df_path=args.in_csv_path,
        transform=transform, mode='test'
        )

    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # json mapping
    f = open('./p2/id2label.json')
    id2label = json.load(f)

    # load unpretrained resnet
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 65)
    model = model.to(device)

    # model
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict['model'])
    print("Finish loading my model...")

    model.eval()

    ids, img_names, labels = [], [], []
    with torch.no_grad():
        for id, img_name, img in test_dl:
            img = img.to(device)
            preds = model(img)
            preds_label = np.argmax(preds.cpu().detach().numpy(), axis=1)
            for id_out, img_name_out, pred_label_out in zip(id, img_name, preds_label):
                ids.append(id_out.item())
                img_names.append(img_name_out)
                labels.append(
                    id2label[str(pred_label_out.item())]
                    )

    df_out = pd.DataFrame(
        {
            'id': ids,
            'filename': img_names,
            'label':labels
        }
    )
    df_out.to_csv(args.out_csv_path, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)