import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.generator import Generator
from utils.metrics import lab2rgb, rgb2lab
import matplotlib.pyplot as plt

def load_model(checkpoint_path, device):
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    return generator

def process_image(image_path, device):
    # Load and preprocess grayscale image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    
    # Load ground truth image (handle modified filenames)
    base_name = os.path.basename(image_path).replace('-modified', '')
    ground_truth_path = os.path.join('ground_truth_images', base_name)
    if os.path.exists(ground_truth_path):
        ground_truth_img = Image.open(ground_truth_path).convert('RGB')
        ground_truth = transform(ground_truth_img)
    else:
        print(f"Warning: Ground truth image not found at {ground_truth_path}")
        ground_truth = img_tensor.clone()
    
    # Convert to LAB
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_lab = rgb2lab(img_np)
    
    # Extract L channel and normalize to [0, 1]
    L = torch.from_numpy(img_lab[:, :, 0]).unsqueeze(0).unsqueeze(0).to(device)
    L = L / 100.0  # Normalize L channel
    
    return L, img_tensor, ground_truth

def colorize_image(generator, L):
    with torch.no_grad():
        fake_ab = generator(L)
        return fake_ab

def visualize_results(original, colorized, ground_truth, save_path=None):
    # Convert to numpy arrays
    original = original.permute(1, 2, 0).numpy()
    ground_truth = ground_truth.permute(1, 2, 0).numpy()
    
    # Create LAB image
    L = rgb2lab(original)[:, :, 0]
    ab = colorized.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab = ab * 255.0 - 128.0  # Scale ab to [-128, 128]
    lab = np.concatenate([L[..., np.newaxis], ab], axis=2)
    
    # Convert to RGB
    colorized_rgb = lab2rgb(lab)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(L, cmap='gray')
    plt.title('Input Grayscale')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(colorized_rgb)
    plt.title('Colorized')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_5.pth', help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    generator = load_model(args.checkpoint, device)
    
    # Process image
    L, original, ground_truth = process_image(args.image_path, device)
    
    # Colorize image
    colorized = colorize_image(generator, L)
    
    # Visualize results
    visualize_results(original, colorized, ground_truth, args.save_path)

if __name__ == '__main__':
    main() 