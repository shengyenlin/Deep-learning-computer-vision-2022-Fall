from torchvision import transforms

def get_mini_transform():
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def get_office_tarnsform(mode):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    elif mode == 'valid':
        transform = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform