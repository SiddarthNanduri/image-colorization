import os
import torch
from test import load_model, process_image, colorize_image, visualize_results
import matplotlib.pyplot as plt
import argparse

def test_all_images(checkpoint_path='checkpoints/checkpoint_5.pth', 
                   input_dir='test_images', 
                   ground_truth_dir='ground_truth_images',
                   output_dir='results'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model from", checkpoint_path)
    generator = load_model(checkpoint_path, device)
    
    # Get list of test images
    test_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nFound {len(test_images)} images to process:")
    
    # Verify ground truth images exist
    for img in test_images:
        gt_name = img.replace('-modified', '')
        gt_path = os.path.join(ground_truth_dir, gt_name)
        status = "✓" if os.path.exists(gt_path) else "✗"
        print(f"- {img} -> {gt_name} [{status}]")
    
    # Process each image
    for img_name in test_images:
        print(f"\nProcessing {img_name}...")
        
        try:
            # Process image
            img_path = os.path.join(input_dir, img_name)
            L, original, ground_truth = process_image(img_path, device)
            
            # Move L to device for colorization
            L = L.to(device)
            
            # Colorize image
            colorized = colorize_image(generator, L)
            
            # Save results
            save_path = os.path.join(output_dir, f'colorized_{img_name}')
            visualize_results(original, colorized, ground_truth, save_path)
            
            print(f"Successfully processed and saved to {save_path}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_5.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, default='test_images',
                      help='Directory containing grayscale test images')
    parser.add_argument('--ground_truth_dir', type=str, default='ground_truth_images',
                      help='Directory containing original color images')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    args = parser.parse_args()
    
    test_all_images(args.checkpoint, args.input_dir, args.ground_truth_dir, args.output_dir) 