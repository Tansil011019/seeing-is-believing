import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class RGANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, img_size=128):
        super().__init__()
        self.init_size = img_size // 16
        self.l1 = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, 128 * self.init_size ** 2)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(16, img_channels, 3, stride=1, padding=1)),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.l1(x)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class RGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=128):
        super().__init__()
        
        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.GroupNorm(1, out_filters),
                nn.CELU(alpha=0.1, inplace=True)
            ]
            return block
        
        self.model = nn.Sequential(
            *conv_block(img_channels, 16, bn=False),
            *conv_block(16, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
        )
        
        self.adv_layer = nn.Linear(128 * (img_size // 16) ** 2, 1)

    def forward(self, x):
        out = self.model(x)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        return validity, out_flat