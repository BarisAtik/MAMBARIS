import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

def load_cifar10(batch_size=64, seed=42):
    """
    Load CIFAR-10 dataset consistently across different runs and notebooks.
    
    Args:
        batch_size (int): Batch size for data loaders
        seed (int): Random seed for reproducibility
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        X_train_tensor: Raw training data tensor
        X_test_tensor: Raw test data tensor
        Y_train_tensor: Training labels tensor
        Y_test_tensor: Test labels tensor
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Define consistent transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Convert to tensors with consistent shape
    X_train = torch.stack([sample[0] for sample in train_dataset])
    Y_train = torch.tensor([sample[1] for sample in train_dataset], dtype=torch.long)
    X_test = torch.stack([sample[0] for sample in test_dataset])
    Y_test = torch.tensor([sample[1] for sample in test_dataset], dtype=torch.long)
    
    # Create datasets
    train_tensor_dataset = TensorDataset(X_train, Y_train)
    test_tensor_dataset = TensorDataset(X_test, Y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
    
    test_loader = DataLoader(
        test_tensor_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle test data
    )
    
    return train_loader, test_loader, X_train, X_test, Y_train, Y_test

def get_class_names():
    """Return the class names for CIFAR-10."""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']
        