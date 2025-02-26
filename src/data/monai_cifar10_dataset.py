import torch
import numpy as np
from torchvision import datasets, transforms
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, RandFlip, RandRotate, NormalizeIntensity, 
    RandSpatialCrop, ScaleIntensity, Transform
)

class TransformWithKeys(Transform):
    """Apply a transform to specific keys in a dictionary."""
    def __init__(self, keys, transform):
        self.keys = keys
        self.transform = transform
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.transform(d[key])
        return d

def load_monai_cifar10_dataset(data_dir, use_augmentation=True, normalize=True, cache_rate=1.0, num_workers=4):
    """
    Load CIFAR-10 dataset with MONAI for faster training.
    
    Args:
        data_dir: Directory where the dataset is stored or will be downloaded
        use_augmentation: Whether to use data augmentation for training
        normalize: Whether to normalize the data
        cache_rate: Percentage of data to be cached (1.0 = all)
        num_workers: Number of workers for initial data loading
        
    Returns:
        train_dataset, val_dataset: MONAI Dataset objects for training and validation
    """
    # First, load the dataset using torchvision to get the raw data
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    # Convert to numpy arrays and ensure channel first format (CIFAR is HWC, we need CHW)
    train_data = cifar_train.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # NHWC -> NCHW and normalize to [0,1]
    train_labels = np.array(cifar_train.targets)
    test_data = cifar_test.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0    # NHWC -> NCHW and normalize to [0,1]
    test_labels = np.array(cifar_test.targets)
    
    # Create data dictionaries for MONAI
    train_dicts = [
        {"image": img, "label": label} 
        for img, label in zip(train_data, train_labels)
    ]
    
    test_dicts = [
        {"image": img, "label": label} 
        for img, label in zip(test_data, test_labels)
    ]
    
    # Define transforms
    train_transforms = []
    test_transforms = []
    
    # Add augmentation if requested
    if use_augmentation:
        train_transforms.extend([
            TransformWithKeys(["image"], RandFlip(prob=0.5, spatial_axis=2)),  # horizontal flip
            TransformWithKeys(["image"], RandRotate(range_x=15, prob=0.5, keep_size=True)),
        ])
    
    # Add normalization if requested
    if normalize:
        train_transforms.append(
            TransformWithKeys(["image"], 
                NormalizeIntensity(subtrahend=[0.4914, 0.4822, 0.4465], divisor=[0.2471, 0.2435, 0.2616], channel_wise=True)
            )
        )
        test_transforms.append(
            TransformWithKeys(["image"], 
                NormalizeIntensity(subtrahend=[0.4914, 0.4822, 0.4465], divisor=[0.2471, 0.2435, 0.2616], channel_wise=True)
            )
        )
    
    # Create transform compositions
    train_transform = Compose(train_transforms)
    test_transform = Compose(test_transforms)
    
    # Create MONAI Datasets
    train_ds = Dataset(
        data=train_dicts,
        transform=train_transform
    )
    
    val_ds = Dataset(
        data=test_dicts,
        transform=test_transform
    )
    
    return train_ds, val_ds


def get_monai_dataloaders(train_ds, val_ds, train_batch_size=128, val_batch_size=256, 
                          num_workers=4, pin_memory=True):
    """
    Create MONAI DataLoaders from Datasets.
    
    Args:
        train_ds: Training Dataset
        val_ds: Validation Dataset
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader: MONAI DataLoaders
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader 