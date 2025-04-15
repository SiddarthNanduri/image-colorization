import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import lab2rgb as skimage_lab2rgb
from skimage.color import rgb2lab as skimage_rgb2lab

def calculate_psnr(img1, img2):
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

def lab_to_rgb(L, ab):
    lab = np.concatenate([L[..., np.newaxis], ab], axis=2)
    return skimage_lab2rgb(lab)

def evaluate_batch(pred_ab, target_ab, L):
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

def lab2rgb(lab):
    return skimage_lab2rgb(lab)

def rgb2lab(rgb):
    return skimage_rgb2lab(rgb) 
