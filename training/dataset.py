from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder("data/processed/train", transform)
    val_dataset = datasets.ImageFolder("data/processed/val", transform)
    test_dataset = datasets.ImageFolder("data/processed/test", transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader