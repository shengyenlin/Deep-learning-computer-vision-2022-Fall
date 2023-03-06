import sys
import os

#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset

#other
from PIL import Image
import pandas as pd


#customized
from preprocess import P1_TransformsModelB, P1TestDataset
from utils import valid

IMG_PATH = sys.argv[1]
PRED_PATH = sys.argv[2]
MODEL_PATH = 'p1_model.pth'
NUM_CLASS = 50

# print(IMG_PATH)
# print(PRED_PATH)

# Since I use torchvision.dataset.ImageFolder to build dataset in training phase,
# and the class is sorted alphabatically (i.e. the class behind '1' is '10' )
# I have to turn the output class into correct label
idx2class = {0: '0',
 1: '1',
 2: '10',
 3: '11',
 4: '12',
 5: '13',
 6: '14',
 7: '15',
 8: '16',
 9: '17',
 10: '18',
 11: '19',
 12: '2',
 13: '20',
 14: '21',
 15: '22',
 16: '23',
 17: '24',
 18: '25',
 19: '26',
 20: '27',
 21: '28',
 22: '29',
 23: '3',
 24: '30',
 25: '31',
 26: '32',
 27: '33',
 28: '34',
 29: '35',
 30: '36',
 31: '37',
 32: '38',
 33: '39',
 34: '4',
 35: '40',
 36: '41',
 37: '42',
 38: '43',
 39: '44',
 40: '45',
 41: '46',
 42: '47',
 43: '48',
 44: '49',
 45: '5',
 46: '6',
 47: '7',
 48: '8',
 49: '9'}

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def main():
    imgs_path = os.listdir(IMG_PATH)
    imgs_path = sorted(imgs_path)
    data_transforms = P1_TransformsModelB()
    test_set = P1TestDataset(
        imgs_path,
        IMG_PATH,
        data_transforms.transform_P1_modelB['val']
        )
    test_loader = DataLoader(
        test_set, batch_size=48, 
        shuffle=False, num_workers=8
    )

    model = models.resnet152()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASS)

    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(
        state_dict['model']
    )

    model = model.to(device)
    preds = []
    model.eval()
    with torch.no_grad():
        for img in test_loader:
            img = img.to(device)
            output = model(img)
            predict = torch.argmax(output, dim=-1)
            predict = predict.detach().cpu().numpy()

            #turn into correct label
            for i in range(len(predict)):
                predict[i] = idx2class[predict[i]]

            preds.extend(predict)

    df = pd.DataFrame(
        []
    )
    df["filename"] = imgs_path
    df["label"] = preds
    df.to_csv(PRED_PATH, index = 0)

if __name__ == '__main__':
    main()