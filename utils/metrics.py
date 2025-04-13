import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import lab2rgb

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

def lab_to_rgb(L, ab):
    """Convert LAB to RGB"""
    L = L.squeeze()  # Remove channel dimension
    L = (L * 100.0)  # Scale to [0, 100]
    ab = (ab * 255.0) - 128.0  # Scale to [-128, 127]
    lab = np.concatenate([L[..., np.newaxis], ab.transpose(1, 2, 0)], axis=2)
    rgb = lab2rgb(lab)
    return rgb

def evaluate_batch(pred_ab, target_ab, L):
    """Evaluate a batch of predictions"""
    batch_size = pred_ab.size(0)
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # Convert to numpy
        pred_ab_np = pred_ab[i].cpu().numpy()
        target_ab_np = target_ab[i].cpu().numpy()
        L_np = L[i].cpu().numpy()
        
        # Convert to RGB
        pred_rgb = lab_to_rgb(L_np, pred_ab_np)
        target_rgb = lab_to_rgb(L_np, target_ab_np)
        
        # Calculate metrics
        psnr_val = calculate_psnr(pred_rgb, target_rgb)
        ssim_val = calculate_ssim(pred_rgb, target_rgb)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    
    return np.mean(psnr_values), np.mean(ssim_values) 