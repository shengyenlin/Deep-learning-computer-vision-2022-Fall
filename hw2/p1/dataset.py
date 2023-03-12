import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

transform = transforms.Compose([
    # transforms.Resize(image_size),
    # transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class P1Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.filenames = sorted(os.listdir(root))
        self.root = root
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, index):
        imagePath = os.path.join(self.root,self.filenames[index])
        image = Image.open(imagePath)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.len