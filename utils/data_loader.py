import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Load COCO dataset
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    self.image_files.append(os.path.join(root, file))
        
        # Split into train/val
        split_idx = int(0.8 * len(self.image_files))
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Convert to numpy and transpose to (H, W, C)
        img_np = img.numpy().transpose(1, 2, 0)
        
        # Convert to LAB color space
        img_lab = rgb2lab(img_np)
        
        # Normalize to [-1, 1]
        img_lab = (img_lab + [0, 128, 128]) / [100, 255, 255]
        img_lab = torch.from_numpy(img_lab.astype(np.float32))
        
        # Split into L and ab channels
        L = img_lab[..., 0].unsqueeze(0)  # Add channel dimension
        ab = img_lab[..., 1:].permute(2, 0, 1)  # Convert to (C, H, W)
        
        return L, ab

def get_dataloader(root_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    train_dataset = ColorizationDataset(root_dir, transform, split='train')
    val_dataset = ColorizationDataset(root_dir, transform, split='val')
    
    # Optimize for MPS
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader 