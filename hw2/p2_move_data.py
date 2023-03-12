import os
import pathlib
import shutil
import pandas as pd

ROOT = './hw2_data/digits/mnistm/data'

def main():
    train_path = os.path.join(ROOT, 'train')
    val_path = os.path.join(ROOT, 'val')
    pathlib.Path(train_path).mkdir(exist_ok=True)
    pathlib.Path(val_path).mkdir(exist_ok=True)
    train_df = pd.read_csv('./hw2_data/digits/mnistm/train.csv')
    val_df = pd.read_csv('./hw2_data/digits/mnistm/val.csv')

    # move training data
    for img in train_df['image_name']:
        shutil.move(
            os.path.join(
                ROOT, str(img)
            ),
            os.path.join(
                train_path, str(img)
            )
        )
    # move val data
    for img in val_df['image_name']:
        shutil.move(
            os.path.join(
                ROOT, str(img)
            ),
            os.path.join(
                val_path, str(img)
            )
        )

if __name__ == '__main__':
    main()