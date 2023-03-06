import os
import json

import pandas as pd 

PATH_TO_DATA = './hw4_data/office/train.csv'
PATH_TO_P2_FILE = './p2/'

def main():
    df_train = pd.read_csv(PATH_TO_DATA)
    label = df_train['label'].unique()
    label = set(label)
    id2label = dict(
        zip(range(len(label)), label)
    )

    label2id = {v: k for k, v in id2label.items()}
    print(id2label)
    print(label2id)

    with open(os.path.join(PATH_TO_P2_FILE, 'label2id.json'), 'w') as f:
        json.dump(label2id, f)

    with open(os.path.join(PATH_TO_P2_FILE, 'id2label.json'), 'w') as f:
        json.dump(id2label, f)

if __name__ == '__main__':
    main()