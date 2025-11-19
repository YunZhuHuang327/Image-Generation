import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic convolutional block with GroupNorm and SiLU activation"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class ResBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    """
    Simplified UNet architecture for DDPM on 28x28 images
    """
    def __init__(self, img_channels=3, time_emb_dim=128, base_dim=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(img_channels, base_dim, 3, padding=1)

        # Downsample - Only 2 levels for 28x28 images
        # 28x28 -> 14x14 -> 7x7
        self.down1 = nn.ModuleList([
            ResBlock(base_dim, base_dim, time_emb_dim),
            ResBlock(base_dim, base_dim, time_emb_dim),
        ])
        self.down1_pool = nn.Conv2d(base_dim, base_dim, 4, 2, 1)

        self.down2 = nn.ModuleList([
            ResBlock(base_dim, base_dim*2, time_emb_dim),
            ResBlock(base_dim*2, base_dim*2, time_emb_dim),
        ])
        self.down2_pool = nn.Conv2d(base_dim*2, base_dim*2, 4, 2, 1)

        # Bottleneck at 7x7
        self.bottleneck = nn.ModuleList([
            ResBlock(base_dim*2, base_dim*4, time_emb_dim),
            ResBlock(base_dim*4, base_dim*4, time_emb_dim),
            ResBlock(base_dim*4, base_dim*2, time_emb_dim),
        ])

        # Upsample 7x7 -> 14x14 -> 28x28
        self.up1 = nn.ConvTranspose2d(base_dim*2, base_dim*2, 4, 2, 1)
        self.up1_blocks = nn.ModuleList([
            ResBlock(base_dim*4, base_dim*2, time_emb_dim),
            ResBlock(base_dim*2, base_dim, time_emb_dim),
        ])

        self.up2 = nn.ConvTranspose2d(base_dim, base_dim, 4, 2, 1)
        self.up2_blocks = nn.ModuleList([
            ResBlock(base_dim*2, base_dim, time_emb_dim),
            ResBlock(base_dim, base_dim, time_emb_dim),
        ])

        # Output projection
        self.out = nn.Conv2d(base_dim, img_channels, 1)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Downsample with skip connections
        skip1 = x
        for block in self.down1:
            x = block(x, t)
        skip1 = x
        x = self.down1_pool(x)

        for block in self.down2:
            x = block(x, t)
        skip2 = x
        x = self.down2_pool(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x, t)

        # Upsample with skip connections
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        for block in self.up1_blocks:
            x = block(x, t)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        for block in self.up2_blocks:
            x = block(x, t)

        # Output
        output = self.out(x)
        return output


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Define beta schedule - register as buffers so they move with the model
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance',
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward diffusion process q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # Forward process (now all tensors are on the same device)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def forward(self, x, t):
        """
        Predict noise
        """
        return self.model(x, t)

    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        # All buffers are already on the same device as the model
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]

        # Call model (current image - noise prediction)
        model_output = self.model(x, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, save_intermediate=False):
        """
        Sample from the model
        """
        device = next(self.model.parameters()).device

        # Start from pure noise
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        imgs = []

        if save_intermediate:
            # Save 8 intermediate steps
            save_steps = torch.linspace(self.timesteps-1, 0, 8).long()

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t)

            if save_intermediate and i in save_steps:
                imgs.append(img.cpu())

        if save_intermediate:
            return img, imgs
        return img
