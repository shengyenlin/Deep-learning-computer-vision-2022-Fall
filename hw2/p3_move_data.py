import os
import sys
import pathlib
import shutil
import pandas as pd

#ROOT = './hw2_data/digits/mnistm/data'
dataset_name = sys.argv[1]

def main():
    root = f'./hw2_data/digits/{dataset_name}/data'
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')
    pathlib.Path(train_path).mkdir(exist_ok=True)
    pathlib.Path(val_path).mkdir(exist_ok=True)
    train_df = pd.read_csv(f'./hw2_data/digits/{dataset_name}/train.csv')
    val_df = pd.read_csv(f'./hw2_data/digits/{dataset_name}/val.csv')

    # move training data
    for img in train_df['image_name']:
        shutil.move(
            os.path.join(
                root, str(img)
            ),
            os.path.join(
                train_path, str(img)
            )
        )
    # move val data
    for img in val_df['image_name']:
        shutil.move(
            os.path.join(
                root, str(img)
            ),
            os.path.join(
                val_path, str(img)
            )
        )

if __name__ == '__main__':
    main()