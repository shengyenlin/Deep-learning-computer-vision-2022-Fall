import sys
import os

import pandas as pd
import numpy as np

GT_PATH = sys.argv[1]
PRED_PATH = sys.argv[2]

def get_label(img_files):
    labels = []
    for img_name in img_files:
        file_name = img_name.split("_")
        label = file_name[0]
        labels.append(int(label))
    return labels

def main():
    df = pd.read_csv(PRED_PATH, header=0)
    preds = np.array(
        df['label']
    )
    

    imgs_path = os.listdir(GT_PATH)
    imgs_path = sorted(imgs_path)
    labels = get_label(imgs_path)
    labels = np.array(labels)

    acc = sum(preds == labels) / preds.shape[0]
    print(f"Recheck acc: {acc}")

if __name__ == '__main__':
    main()