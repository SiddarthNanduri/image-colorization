import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.generator import Generator
from utils.metrics import lab_to_rgb, calculate_psnr, calculate_ssim

def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    generator = Generator().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Process images
    psnr_values = []
    ssim_values = []
    
    for img_name in tqdm(os.listdir(args.test_dir)):
        if not img_name.endswith(('.jpg', '.png')):
            continue
            
        # Load and preprocess image
        img_path = os.path.join(args.test_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        
        # Convert to LAB
        img_np = np.array(img)
        img_lab = rgb2lab(img_np)
        img_lab = (img_lab + 128) / 255.0
        img_lab = torch.from_numpy(img_lab).float()
        
        L = img_lab[[0], ...].unsqueeze(0).to(device)
        ab = img_lab[[1, 2], ...].unsqueeze(0).to(device)
        
        # Generate colorization
        with torch.no_grad():
            fake_ab = generator(L)
        
        # Convert to RGB
        L_np = L[0].cpu().numpy()
        fake_ab_np = fake_ab[0].cpu().numpy()
        target_ab_np = ab[0].cpu().numpy()
        
        fake_rgb = lab_to_rgb(L_np, fake_ab_np)
        target_rgb = lab_to_rgb(L_np, target_ab_np)
        
        # Calculate metrics
        psnr = calculate_psnr(fake_rgb, target_rgb)
        ssim = calculate_ssim(fake_rgb, target_rgb)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        
        # Save results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(fake_rgb)
        plt.title('Colorized')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(target_rgb)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.savefig(os.path.join(args.output_dir, f'result_{img_name}'))
        plt.close()
    
    # Print metrics
    print(f'Average PSNR: {np.mean(psnr_values):.2f}')
    print(f'Average SSIM: {np.mean(ssim_values):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    evaluate(args) 