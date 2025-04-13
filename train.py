import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_dataloader
from utils.metrics import evaluate_batch

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Create optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    # Get data loaders
    train_loader, val_loader = get_dataloader(args.data_dir, args.batch_size)
    
    # Tensorboard writer
    writer = SummaryWriter()
    
    # Training loop
    for epoch in range(args.epochs):
        # Training
        generator.train()
        discriminator.train()
        
        train_g_loss = 0
        train_d_loss = 0
        train_psnr = 0
        train_ssim = 0
        
        for i, (L, ab) in enumerate(tqdm(train_loader)):
            L = L.to(device)
            ab = ab.to(device)
            batch_size = L.size(0)
            
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_images = torch.cat([L, ab], dim=1)
            real_labels = torch.ones(batch_size, 1, 10, 10).to(device)  # Updated size to match discriminator output
            d_real = discriminator(real_images)
            d_real_loss = criterion_gan(d_real, real_labels)
            
            # Fake images
            fake_ab = generator(L)
            fake_images = torch.cat([L, fake_ab], dim=1)
            fake_labels = torch.zeros(batch_size, 1, 10, 10).to(device)  # Updated size to match discriminator output
            d_fake = discriminator(fake_images.detach())
            d_fake_loss = criterion_gan(d_fake, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            if i % 5 == 0:  # Update generator less frequently
                g_optimizer.zero_grad()
                
                # GAN loss
                g_fake = discriminator(fake_images)
                g_loss_gan = criterion_gan(g_fake, real_labels)
                
                # L1 loss
                g_loss_l1 = criterion_l1(fake_ab, ab)
                
                g_loss = g_loss_gan + 100 * g_loss_l1
                g_loss.backward()
                g_optimizer.step()
                
                train_g_loss += g_loss.item()
                train_d_loss += d_loss.item()
                
                # Calculate metrics
                psnr, ssim = evaluate_batch(fake_ab.detach(), ab, L)
                train_psnr += psnr
                train_ssim += ssim
        
        # Validation
        generator.eval()
        val_g_loss = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for L, ab in val_loader:
                L = L.to(device)
                ab = ab.to(device)
                
                fake_ab = generator(L)
                g_loss_l1 = criterion_l1(fake_ab, ab)
                val_g_loss += g_loss_l1.item()
                
                psnr, ssim = evaluate_batch(fake_ab, ab, L)
                val_psnr += psnr
                val_ssim += ssim
        
        # Log metrics
        writer.add_scalar('Loss/train_generator', train_g_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/train_discriminator', train_d_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/validation', val_g_loss / len(val_loader), epoch)
        writer.add_scalar('Metrics/train_psnr', train_psnr / len(train_loader), epoch)
        writer.add_scalar('Metrics/train_ssim', train_ssim / len(train_loader), epoch)
        writer.add_scalar('Metrics/val_psnr', val_psnr / len(val_loader), epoch)
        writer.add_scalar('Metrics/val_ssim', val_ssim / len(val_loader), epoch)
        
        print(f'Epoch [{epoch+1}/{args.epochs}]')
        print(f'Train - G_Loss: {train_g_loss/len(train_loader):.4f}, D_Loss: {train_d_loss/len(train_loader):.4f}')
        print(f'Val - Loss: {val_g_loss/len(val_loader):.4f}, PSNR: {val_psnr/len(val_loader):.2f}, SSIM: {val_ssim/len(val_loader):.4f}')
        
        # Save checkpoints
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, f'checkpoints/checkpoint_{epoch+1}.pth')
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to COCO dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval for checkpoints')
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    
    train(args) 