import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CIFAR10Dataset(Dataset):
    """
    Custom CIFAR-10 dataset that works with MONAI transforms.
    """
    def __init__(self, data_dir, train=True, use_augmentation=True, normalize=True):
        # Load CIFAR-10 dataset
        cifar_dataset = datasets.CIFAR10(root=data_dir, train=train, download=True)
        
        # Convert to numpy arrays and ensure channel first format (CIFAR is HWC, we need CHW)
        self.data = cifar_dataset.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # NHWC -> NCHW
        self.targets = np.array(cifar_dataset.targets)
        
        # Define transforms
        transform_list = []
        
        # Add augmentation if requested and if training
        if use_augmentation and train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
        
        # Add normalization if requested
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2471, 0.2435, 0.2616]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx])
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_monai_cifar10_dataset(data_dir, use_augmentation=True, normalize=True, cache_rate=1.0, num_workers=4):
    """
    Load CIFAR-10 dataset with PyTorch transforms for faster training.
    
    Args:
        data_dir: Directory where the dataset is stored or will be downloaded
        use_augmentation: Whether to use data augmentation for training
        normalize: Whether to normalize the data
        cache_rate: Not used, kept for API compatibility
        num_workers: Not used, kept for API compatibility
        
    Returns:
        train_dataset, val_dataset: Dataset objects for training and validation
    """
    train_dataset = CIFAR10Dataset(
        data_dir=data_dir,
        train=True,
        use_augmentation=use_augmentation,
        normalize=normalize
    )
    
    val_dataset = CIFAR10Dataset(
        data_dir=data_dir,
        train=False,
        use_augmentation=False,
        normalize=normalize
    )
    
    return train_dataset, val_dataset


def get_monai_dataloaders(train_ds, val_ds, train_batch_size=128, val_batch_size=256, 
                          num_workers=4, pin_memory=True):
    """
    Create DataLoaders from Datasets.
    
    Args:
        train_ds: Training Dataset
        val_ds: Validation Dataset
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader: DataLoaders
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