import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _balanced_subset(dataset, samples_per_class: int, seed: int = 42) -> Subset:
    """
    Return a Subset with exactly `samples_per_class` examples from each class,
    chosen randomly with a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    indices_by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.targets):
        indices_by_class.setdefault(label, []).append(idx)

    chosen = []
    for label, indices in sorted(indices_by_class.items()):
        k = min(samples_per_class, len(indices))
        chosen.extend(rng.sample(indices, k))

    rng.shuffle(chosen)
    return Subset(dataset, chosen)


def get_dataloaders(num_workers: int = 4, train_samples_per_class: int = 20_000):
    """
    train_samples_per_class: how many images per class to use for training.
    Set to None to use the full training set.
    Val and test sets are always used in full.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder("140k-faces/real_vs_fake/real-vs-fake/train", train_transform)
    val_dataset   = datasets.ImageFolder("140k-faces/real_vs_fake/real-vs-fake/valid", eval_transform)
    test_dataset  = datasets.ImageFolder("140k-faces/real_vs_fake/real-vs-fake/test",  eval_transform)

    if train_samples_per_class is not None:
        train_dataset = _balanced_subset(train_dataset, train_samples_per_class)
        total = train_samples_per_class * 2
        print(f"Using {train_samples_per_class:,} samples/class  →  {total:,} training images total")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader