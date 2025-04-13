import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Inception-ResNet-v2 feature extractor
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.aux_logits = False  # Disable auxiliary logits
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 2048, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Convert single channel to 3 channels for Inception
        x_inception = x.repeat(1, 3, 1, 1)
        
        # Encoder
        x_encoded = self.encoder(x)
        
        # Inception features
        x_inception = self.inception(x_inception)
        x_inception = x_inception.view(x_inception.size(0), -1, 1, 1)
        x_inception = x_inception.expand(-1, -1, x_encoded.size(2), x_encoded.size(3))
        
        # Fusion
        x_fused = torch.cat([x_encoded, x_inception], dim=1)
        x_fused = self.fusion(x_fused)
        
        # Decoder
        x_decoded = self.decoder(x_fused)
        
        # Ensure output size matches input size
        if x_decoded.size() != (x.size(0), 2, x.size(2), x.size(3)):
            x_decoded = F.interpolate(x_decoded, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        
        return x_decoded 