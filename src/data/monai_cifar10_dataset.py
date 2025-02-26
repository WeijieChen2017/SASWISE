import torch
import numpy as np
from torchvision import datasets, transforms
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, ToTensor, RandFlip, RandRotate, NormalizeIntensity, 
    RandSpatialCrop, ScaleIntensity, EnsureChannelFirst
)

def load_monai_cifar10_dataset(data_dir, use_augmentation=True, normalize=True, cache_rate=1.0, num_workers=4):
    """
    Load CIFAR-10 dataset with MONAI caching for faster training.
    
    Args:
        data_dir: Directory where the dataset is stored or will be downloaded
        use_augmentation: Whether to use data augmentation for training
        normalize: Whether to normalize the data
        cache_rate: Percentage of data to be cached (1.0 = all)
        num_workers: Number of workers for initial data loading
        
    Returns:
        train_dataset, val_dataset: MONAI CacheDataset objects for training and validation
    """
    # First, load the dataset using torchvision to get the raw data
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    # Convert to numpy arrays
    train_data = cifar_train.data  # Shape: (50000, 32, 32, 3)
    train_labels = np.array(cifar_train.targets)
    test_data = cifar_test.data    # Shape: (10000, 32, 32, 3)
    test_labels = np.array(cifar_test.targets)
    
    # Create data dictionaries for MONAI
    train_dicts = [
        {"image": img.astype(np.float32), "label": label} 
        for img, label in zip(train_data, train_labels)
    ]
    
    test_dicts = [
        {"image": img.astype(np.float32), "label": label} 
        for img, label in zip(test_data, test_labels)
    ]
    
    # Define transforms
    train_transforms = []
    test_transforms = []
    
    # Always ensure channel first (MONAI expects this)
    train_transforms.append(EnsureChannelFirst())
    test_transforms.append(EnsureChannelFirst())
    
    # Add augmentation if requested
    if use_augmentation:
        train_transforms.extend([
            RandFlip(prob=0.5, spatial_axis=1),  # horizontal flip
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandSpatialCrop(roi_size=(32, 32), random_size=False),
        ])
    
    # Scale intensities to [0, 1]
    train_transforms.append(ScaleIntensity(minv=0.0, maxv=1.0))
    test_transforms.append(ScaleIntensity(minv=0.0, maxv=1.0))
    
    # Add normalization if requested
    if normalize:
        train_transforms.append(
            NormalizeIntensity(subtrahend=[0.4914, 0.4822, 0.4465], divisor=[0.2471, 0.2435, 0.2616])
        )
        test_transforms.append(
            NormalizeIntensity(subtrahend=[0.4914, 0.4822, 0.4465], divisor=[0.2471, 0.2435, 0.2616])
        )
    
    # Create transform compositions
    train_transform = Compose(train_transforms)
    test_transform = Compose(test_transforms)
    
    # Create MONAI CacheDatasets
    train_ds = CacheDataset(
        data=train_dicts,
        transform=train_transform,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    
    val_ds = CacheDataset(
        data=test_dicts,
        transform=test_transform,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    
    return train_ds, val_ds


def get_monai_dataloaders(train_ds, val_ds, train_batch_size=128, val_batch_size=256, 
                          num_workers=4, pin_memory=True):
    """
    Create MONAI DataLoaders from CacheDatasets.
    
    Args:
        train_ds: Training CacheDataset
        val_ds: Validation CacheDataset
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