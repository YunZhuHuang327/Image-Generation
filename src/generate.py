import torch
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
from model import UNet, DDPM
import numpy as np
from PIL import Image


def generate_images(args):
    """
    Generate images using trained DDPM model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Load checkpoint
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    total_images = args.num_images
    num_batches = (total_images + args.batch_size - 1) // args.batch_size
    image_count = 0

    print(f"Generating {total_images} images...")

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Calculate batch size for the last batch
            current_batch_size = min(args.batch_size, total_images - image_count)

            # Generate samples
            samples = ddpm.sample(
                image_size=args.image_size,
                batch_size=current_batch_size,
                channels=3
            )

            # Denormalize from [-1, 1] to [0, 1]
            samples = (samples.clamp(-1, 1) + 1) / 2

            # Save each image individually
            for i in range(current_batch_size):
                image_count += 1
                # Convert to PIL Image and save
                img_tensor = samples[i]
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_array = np.transpose(img_array, (1, 2, 0))  # CHW to HWC
                img_pil = Image.fromarray(img_array)

                # Save with zero-padded filename
                filename = f"{image_count:05d}.png"
                filepath = os.path.join(args.output_dir, filename)
                img_pil.save(filepath)

            # Optionally save a grid of samples
            if batch_idx == 0:
                grid_path = os.path.join(args.output_dir, 'sample_grid.png')
                save_image(samples[:min(16, current_batch_size)], grid_path, nrow=4)
                print(f"\nSample grid saved to {grid_path}")

    print(f"\nSuccessfully generated {image_count} images in {args.output_dir}")


def visualize_diffusion_process(args):
    """
    Visualize the diffusion process by generating images at intermediate steps
    This is for the report requirement
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Load checkpoint
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate 8 samples with intermediate steps
    num_samples = 8
    num_intermediate_steps = 7  # We want 8 total images per sample (including final)

    # Calculate which timesteps to save
    save_steps = torch.linspace(args.timesteps-1, 0, num_intermediate_steps+1).long().tolist()

    print(f"Generating {num_samples} samples with intermediate steps...")
    print(f"Saving at timesteps: {save_steps}")

    all_images = []

    with torch.no_grad():
        # Start from pure noise
        img = torch.randn((num_samples, 3, args.image_size, args.image_size), device=device)

        for i in tqdm(reversed(range(0, args.timesteps)), desc="Diffusion steps"):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            # Denoise one step
            betas_t = ddpm.betas[t][:, None, None, None].to(device)
            sqrt_one_minus_alphas_cumprod_t = ddpm.sqrt_one_minus_alphas_cumprod[t][:, None, None, None].to(device)
            sqrt_recip_alphas_t = ddpm.sqrt_recip_alphas[t][:, None, None, None].to(device)

            # Call model
            model_output = ddpm.model(img, t)
            model_mean = sqrt_recip_alphas_t * (
                img - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
            )

            if t[0] == 0:
                img = model_mean
            else:
                posterior_variance_t = ddpm.posterior_variance[t][:, None, None, None].to(device)
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise

            # Save intermediate results
            if i in save_steps:
                # Denormalize
                img_to_save = (img.clamp(-1, 1) + 1) / 2
                all_images.append(img_to_save.cpu())

    # Stack all images: [num_steps, num_samples, C, H, W]
    all_images = torch.stack(all_images)
    print(f"All images shape: {all_images.shape}")

    # Rearrange to: [num_samples, num_steps, C, H, W]
    all_images = all_images.permute(1, 0, 2, 3, 4)

    # Flatten to create grid: [num_samples * num_steps, C, H, W]
    all_images_flat = all_images.reshape(-1, 3, args.image_size, args.image_size)

    # Save as grid (num_samples rows x num_steps columns)
    grid_path = os.path.join(args.output_dir, 'diffusion_process.png')
    save_image(all_images_flat, grid_path, nrow=num_intermediate_steps+1)
    print(f"\nDiffusion process visualization saved to {grid_path}")

    # Also save individual steps
    for step_idx in range(num_intermediate_steps+1):
        step_images = all_images[:, step_idx]
        step_path = os.path.join(args.output_dir, f'step_{step_idx}.png')
        save_image(step_images, step_path, nrow=4)


def main():
    parser = argparse.ArgumentParser(description='Generate images using trained DDPM')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./generated_images',
                        help='Output directory for generated images')
    parser.add_argument('--image_size', type=int, default=28, help='Image size')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value')
    parser.add_argument('--time_emb_dim', type=int, default=128, help='Time embedding dimension')
    parser.add_argument('--base_dim', type=int, default=64, help='Base dimension for UNet')

    # Generation parameters
    parser.add_argument('--num_images', type=int, default=10000,
                        help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for generation')

    # Mode selection
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'visualize'],
                        help='Mode: generate images or visualize diffusion process')

    args = parser.parse_args()

    # Print configuration
    print("="*50)
    print("Generation Configuration:")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50)

    if args.mode == 'generate':
        generate_images(args)
    elif args.mode == 'visualize':
        visualize_diffusion_process(args)


if __name__ == '__main__':
    main()
