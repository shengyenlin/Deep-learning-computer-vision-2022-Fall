from torchvision import transforms

class P2Transformation:
    def __init__(self):
        self.random_erase = 0.1
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        
        self.transform= {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees = (-20, 20)),
                transforms.ToTensor(),
                self.normalize,
                # transforms.RandomErasing(p=self.random_erase),
            ]),
            'val':  transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ])
        }