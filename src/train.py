import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
from model import UNet, DDPM
from PIL import Image


class MNISTDataset(Dataset):
    """Custom dataset for MNIST images in a flat directory"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Get all png files
        self.image_files = sorted([f for f in os.listdir(data_path) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label


def get_data_loader(data_path, batch_size, image_size=28):
    """
    Create data loader for the provided RGB MNIST dataset
    """
    # Transform for RGB images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1] for RGB
    ])

    # Use custom dataset to load images from flat directory
    dataset = MNISTDataset(
        data_path=data_path,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def train(args):
    """
    Training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loader
    dataloader = get_data_loader(args.data_path, args.batch_size, args.image_size)
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Create model
    unet = UNet(
        img_channels=3,
        time_emb_dim=args.time_emb_dim,
        base_dim=args.base_dim
    )

    ddpm = DDPM(
        model=unet,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Loss function
    criterion = nn.MSELoss()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Training loop
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        ddpm.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()

            # Forward diffusion: add noise to images
            noise = torch.randn_like(images)
            x_t, noise = ddpm.forward_diffusion(images, t, noise)

            # Predict noise
            predicted_noise = ddpm(x_t, t)

            # Calculate loss
            loss = criterion(predicted_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': epoch_loss / (batch_idx + 1),
                'lr': optimizer.param_groups[0]['lr']
            })

            # Save sample images periodically
            if global_step % args.sample_interval == 0:
                ddpm.eval()
                with torch.no_grad():
                    samples = ddpm.sample(
                        image_size=args.image_size,
                        batch_size=16,
                        channels=3
                    )
                    samples = (samples.clamp(-1, 1) + 1) / 2  # Denormalize to [0, 1]
                    save_image(
                        samples,
                        os.path.join(args.sample_dir, f'sample_step_{global_step}.png'),
                        nrow=4
                    )
                ddpm.train()

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{args.epochs} - Average Loss: {avg_epoch_loss:.6f}")

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'ddpm_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"Best model saved with loss: {best_loss:.6f}")

    print("Training completed!")

    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/mnist_rgb',
                        help='Path to RGB MNIST dataset (ImageFolder format)')
    parser.add_argument('--image_size', type=int, default=28, help='Image size')

    # Model parameters
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value')
    parser.add_argument('--time_emb_dim', type=int, default=128, help='Time embedding dimension')
    parser.add_argument('--base_dim', type=int, default=64, help='Base dimension for UNet')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')

    # Checkpoint and sampling
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--sample_dir', type=str, default='./samples', help='Sample directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval (epochs)')
    parser.add_argument('--sample_interval', type=int, default=500, help='Sample generation interval (steps)')

    args = parser.parse_args()

    # Print configuration
    print("="*50)
    print("Training Configuration:")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50)

    train(args)


if __name__ == '__main__':
    main()
