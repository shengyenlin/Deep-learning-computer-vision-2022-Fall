import sys
import random

import pandas as pd

#torch
import torch

from p3.model import DANN
from p3.dataset import P3TestDataset, get_transform

# Set random seed for reproducibility
SEED = 5566
#manualSeed = random.randint(1, 10000) # use if you want new results
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IMG_DIR = sys.argv[1]
OUT_PATH = sys.argv[2]

class Args:
    def __init__(self):
        self.image_size = 28
        self.channels = 3
        self.batch_size = 128
        self.use_dann = 1

def main():
    args = Args()
    transform = get_transform(args)
    test_ds = P3TestDataset(TEST_IMG_DIR, transform=transform)
    test_dl = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False
        )
    
    model = DANN(args.use_dann).to(device)

    if 'svhn' in TEST_IMG_DIR:
        model_path = 'p3_svhn.ckpt'
    elif 'usps' in TEST_IMG_DIR:
        model_path = 'p3_usps.ckpt'

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model'])
    
    model.eval()
    lam = 0 #we don't need to compute domain clf in eval
    rows = []
    with torch.no_grad():
        for img_names, img in test_dl:
            img = img.to(device)
            class_out, _ = model(img, lam)
            pred_class = torch.max(class_out, axis=1)[1]
            for img_name, pred in zip(img_names, pred_class.detach().cpu().numpy()):
                rows.append(
                    {
                        "image_name": img_name,
                        "label": pred
                    }
                )
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

if __name__ == '__main__':
    main()